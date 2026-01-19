#!/usr/bin/env python3
"""
XML utilities for TagSpeech (multi-speaker diarization and ASR).
Handles XML format construction and parsing.
"""

import re
import json
import xml.etree.ElementTree as ET
from typing import List, Dict, Any, Optional
from dataclasses import dataclass


@dataclass
class SpeakerSegment:
    """Represents a speaker segment with timing and text."""
    speaker_id: str
    start_time: float
    end_time: float
    text: str
    gender: str = "unknown"


def construct_multi_speaker_xml(cut) -> str:
    """Construct multi-speaker XML (decoupled text and speaker info).
    
    Separates semantic (text) and speaker information for better structure.
    
    Format:
        <text>
        0.00-3.14>TEXT HERE
        2.00-4.50>OVERLAPPING TEXT
        3.14-5.20>MORE TEXT
        </text>
        <speaker>
        <spk id="1" g="m" t="0.00-3.14,3.14-5.20"/>
        <spk id="2" g="f" t="2.00-4.50"/>
        </speaker>
    
    Args:
        cut: Lhotse Cut object with multiple supervisions
        
    Returns:
        XML string
    """
    # Collect all segments with time and speaker info
    segments = []
    speaker_info = {}  # speaker_id -> (gender, time_ranges)
    
    for sup in cut.supervisions:
        start_time = sup.start
        end_time = sup.start + sup.duration
        text = sup.text.strip()
        speaker_id = sup.speaker
        gender = getattr(sup, 'gender', 'unknown')
        
        segments.append({
            'start': start_time,
            'end': end_time,
            'text': text,
            'speaker': speaker_id,
            'gender': gender,
        })
        
        # Track speaker info
        if speaker_id not in speaker_info:
            speaker_info[speaker_id] = {
                'gender': gender,
                'time_ranges': [],
                'first_appearance': start_time,
            }
        speaker_info[speaker_id]['time_ranges'].append((start_time, end_time))
    
    # Sort segments by start time for text section
    segments.sort(key=lambda x: x['start'])
    
    # Build text section
    text_parts = ['<text>']
    for seg in segments:
        text_parts.append(f"{seg['start']:.2f}-{seg['end']:.2f}>{seg['text']}")
    text_parts.append('</text>')
    
    # Build speakers section
    # Sort speakers by first appearance
    sorted_speakers = sorted(speaker_info.items(), key=lambda x: x[1]['first_appearance'])
    
    speaker_parts = ['<speaker>']
    for speaker_idx, (original_speaker_id, info) in enumerate(sorted_speakers, 1):
        gender = info['gender']
        # Convert gender to single letter
        if gender == 'M':
            gender_short = 'm'
        elif gender == 'F':
            gender_short = 'f'
        else:
            gender_short = 'u'
        
        # Sort time ranges and format as comma-separated list
        time_ranges = sorted(info['time_ranges'], key=lambda x: x[0])
        time_str = ','.join([f"{start:.2f}-{end:.2f}" for start, end in time_ranges])
        
        speaker_parts.append(f'<spk id="{speaker_idx}" g="{gender_short}" t="{time_str}"/>')
    speaker_parts.append('</speaker>')
    
    return '\n'.join(text_parts + speaker_parts)


def parse_xml_to_segments(xml_text: str) -> List[SpeakerSegment]:
    """Parse XML output into speaker segments.
    
    Supports the format (decoupled text and speakers):
        <text>
        0.00-3.14>TEXT HERE
        2.00-4.50>OVERLAPPING TEXT
        </text>
        <speaker>
        <spk id="1" g="m" t="0.00-3.14,3.14-5.20"/>
        <spk id="2" g="f" t="2.00-4.50"/>
        </speaker>
    
    Args:
        xml_text: XML string from model output
        
    Returns:
        List of SpeakerSegment objects
    """
    segments = []
    
    try:
        # Clean up the XML text
        xml_text = xml_text.strip()
        
        # Auto-fix incomplete XML from model truncation
        # If XML doesn't end with </speaker>, try adding it
        if not xml_text.endswith('</speaker>'):
            # Check if there's an incomplete closing tag
            if xml_text.endswith('</speaker'):
                xml_text += '>'
            # Check if there's a <speaker> section that wasn't closed
            elif '<speaker>' in xml_text and '</speaker>' not in xml_text:
                xml_text += '\n</speaker>'
        
        # Similarly for </text>
        if '<text>' in xml_text and not xml_text.endswith('</text>') and '</text>' not in xml_text:
            # Find where text section should end (before <speaker> if it exists)
            if '<speaker>' in xml_text:
                speaker_pos = xml_text.find('<speaker>')
                xml_text = xml_text[:speaker_pos] + '</text>\n' + xml_text[speaker_pos:]
        
        # Parse XML
        root = ET.fromstring(f"<root>{xml_text}</root>")
        
        # Parse text section to get time->text mapping
        text_map = {}  # (start, end) -> text
        text_elem = root.find('text')
        if text_elem is not None:
            text_content = text_elem.text or ""
            lines = text_content.split('\n')
            pattern = r'([0-9.]+)-([0-9.]+)>(.+)'
            
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                match = re.match(pattern, line)
                if match:
                    start_time = float(match.group(1))
                    end_time = float(match.group(2))
                    text = match.group(3).strip()
                    text_map[(start_time, end_time)] = text
        
        # Parse speaker section
        speaker_elem = root.find('speaker')
        if speaker_elem is not None:
            # Find all <spk> child elements
            for child in speaker_elem.findall('spk'):
                speaker_id = child.get('id', 'unknown')
                gender = child.get('g', 'unknown')
                time_str = child.get('t', '')
                
                # Convert short gender to full form
                gender_map = {'m': 'male', 'f': 'female', 'u': 'unknown'}
                if gender in gender_map:
                    gender = gender_map[gender]
                
                # Parse time ranges: "0.00-3.14,3.14-5.20"
                time_ranges = time_str.split(',')
                for time_range in time_ranges:
                    time_range = time_range.strip()
                    if not time_range:
                        continue
                    
                    parts = time_range.split('-')
                    if len(parts) == 2:
                        start_time = float(parts[0])
                        end_time = float(parts[1])
                        
                        # Get text from text_map
                        text = text_map.get((start_time, end_time), "")
                        
                        segments.append(SpeakerSegment(
                            speaker_id=speaker_id,
                            start_time=start_time,
                            end_time=end_time,
                            text=text,
                            gender=gender
                        ))
        
    except ET.ParseError as e:
        print(f"XML parsing error: {e}")
        return []
    except Exception as e:
        print(f"Error parsing XML: {e}")
        return []
    
    # Sort by start time
    segments.sort(key=lambda x: x.start_time)
    
    return segments


def xml_to_json(xml_text: str, pretty: bool = True) -> Optional[str]:
    """Convert XML output to JSON format for easier reading.
    
    Converts TagSpeech XML format to a structured JSON with segments.
    Each segment includes: start, end, text, speaker_id, speaker_gender.
    
    Args:
        xml_text: XML string from model output
        pretty: If True, return formatted JSON with indentation
        
    Returns:
        JSON string if XML is valid, None if parsing failed
        
    Example output:
        {
          "segments": [
            {
              "start": 0.0,
              "end": 3.14,
              "text": "Hello world",
              "speaker_id": "1",
              "speaker_gender": "male"
            },
            ...
          ]
        }
    """
    # Parse XML to segments
    segments = parse_xml_to_segments(xml_text)
    
    if not segments:
        # Return empty result if parsing failed
        return None
    
    # Convert segments to JSON-serializable format
    json_data = {
        "segments": [
            {
                "start": seg.start_time,
                "end": seg.end_time,
                "text": seg.text,
                "speaker_id": seg.speaker_id,
                "speaker_gender": seg.gender,
            }
            for seg in segments
        ]
    }
    
    # Convert to JSON string
    if pretty:
        return json.dumps(json_data, indent=2, ensure_ascii=False)
    else:
        return json.dumps(json_data, ensure_ascii=False)
