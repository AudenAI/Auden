#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Single-node 8-GPU vLLM batch inference script (spawn multi-process version)
Input: jsonl.scp (each line is a JSON containing idx/path/duration/detected_format)
Output: output/<idx>.json
"""

import os
os.environ['VLLM_USE_V1'] = '0'
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import json
import sys
import signal
import psutil
import torch
import warnings
import argparse
from pathlib import Path
from tqdm import tqdm
from vllm import LLM, SamplingParams
import torch.multiprocessing as mp

warnings.filterwarnings('ignore')


def kill_all_children(pid=None, including_parent=False):
    pid = pid or os.getpid()
    parent = psutil.Process(pid)
    children = parent.children(recursive=True)
    for p in children:
        try:
            p.kill()
        except psutil.NoSuchProcess:
            pass
    gone, alive = psutil.wait_procs(children, timeout=3)
    for p in alive:
        try:
            p.send_signal(signal.SIGKILL)
        except psutil.NoSuchProcess:
            pass
    if including_parent:
        try:
            parent.kill()
            parent.wait(3)
        except psutil.NoSuchProcess:
            pass

def clean_exit(signum, frame):
    print('\n[!] Ctrl-C caught, cleaning up ...')
    kill_all_children()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print('[!] All children killed & GPU memory released.')
    os._exit(0)

signal.signal(signal.SIGINT, clean_exit)


def persistent_worker_fn(rank, task_q, result_q, model_path):
    # The processor is loaded once inside the child process and reused later.
    from modelscope import Qwen3OmniMoeProcessor
    from qwen_omni_utils import process_mm_info
    proc = Qwen3OmniMoeProcessor.from_pretrained(model_path)
    print(f'[Worker-{rank}] Processor loaded, ready to work.')

    while True:
        item = task_q.get()
        if item == 'STOP':  
            result_q.put((None, None)) 
            print(f'[Worker-{rank}] Received STOP, exiting.')
            break
        
        try:
            messages = [{"role": "user", "content": [{"type": "audio", "audio": item["path"]}]}]
            text = proc.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            audios, _, _ = process_mm_info(messages, use_audio_in_video=False)

            # add check
            if any(len(a) == 0 for a in audios):
                raise ValueError(f"Zero-length audio waveform detected in {item['path']}")

            result_q.put((item, {
                'prompt': text,
                'multi_modal_data': {'audio': audios} if audios else {}
            }))
        except Exception as e:
            print(f'[Worker-{rank}] Error processing {item.get("idx", "?")}: {e}')
            result_q.put((item, None)) 


def scp_iter(scp_path, batch_size):
    batch = []
    with open(scp_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            batch.append(json.loads(line))
            if len(batch) == batch_size:
                yield batch
                batch = []
    if batch:
        yield batch


def main():
    parser = argparse.ArgumentParser(description='8GPU vLLM batch inference script')
    parser.add_argument('--model_path', type=str, required=True,
                        help='e.g. /path/to/Qwen3-Omni-30B-A3B-Captioner')
    parser.add_argument('--scp_file', type=str, required=True,
                        help='e.g. jsonl.scp')
    parser.add_argument('--dataset_size', type=int, required=True)
    parser.add_argument('--out_dir', type=str, required=True)
    parser.add_argument('--num_workers', type=int, default=196)
    parser.add_argument('--queue_max', type=int, default=4096)
    parser.add_argument('--max_seqs', type=int, default=8)
    
    args = parser.parse_args()
    
    MODEL_PATH = args.model_path
    SCP_FILE = args.scp_file
    DATASET_SIZE = args.dataset_size
    OUT_DIR = Path(args.out_dir)
    OUT_DIR.mkdir(exist_ok=True, parents=True)
    NUM_WORKERS = args.num_workers
    QUEUE_MAX = args.queue_max
    max_seqs = args.max_seqs
    
    print(f'[Config] MODEL_PATH: {MODEL_PATH}')
    print(f'[Config] SCP_FILE: {SCP_FILE}')
    print(f'[Config] DATASET_SIZE: {DATASET_SIZE}')
    print(f'[Config] OUT_DIR: {OUT_DIR}')
    print(f'[Config] NUM_WORKERS: {NUM_WORKERS}')
    print(f'[Config] QUEUE_MAX: {QUEUE_MAX}')
    print(f'[Config] max_seqs: {max_seqs}')
    
    mp.set_start_method("spawn", force=True)
    task_q = mp.Queue(maxsize=QUEUE_MAX)     
    result_q = mp.Queue(maxsize=QUEUE_MAX)  


    print(f'[Main] Starting {NUM_WORKERS} persistent workers...')
    procs = []
    for i in range(NUM_WORKERS):
        p = mp.Process(target=persistent_worker_fn, args=(i, task_q, result_q, MODEL_PATH))
        p.start()
        procs.append(p)
    print(f'[Main] All workers started.')

    # ========== Intialize vLLM ==========
    llm = LLM(
        model=MODEL_PATH,
        trust_remote_code=True,
        gpu_memory_utilization=0.9,
        tensor_parallel_size=8,
        limit_mm_per_prompt={'audio': 1},
        max_num_seqs=max_seqs,
        max_model_len=8192,
        seed=1234,
        dtype="bfloat16",
    )
    sampling_params = SamplingParams(
        temperature=0.8, top_p=0.9, top_k=20, max_tokens=8192
    )


    for b_idx, batch in enumerate(scp_iter(SCP_FILE, DATASET_SIZE), 1):
        print(f'>>> batch {b_idx}, size={len(batch)}')
        
        for item in batch:
            task_q.put(item)
        
        cache = []
        pbar = tqdm(total=len(batch), desc=f'GPU batch {b_idx}')
        processed = 0
        
        while processed < len(batch) or cache:
            while processed < len(batch) and len(cache) < max_seqs * 2:
                try:
                    item, vllm_input = result_q.get_nowait() 
                    if item is not None and vllm_input is not None:
                        cache.append((item, vllm_input))
                    processed += 1
                except:
                    break 
            
            if len(cache) >= max_seqs:
                inference_batch = cache[:max_seqs]
                cache = cache[max_seqs:]
                
                outputs = llm.generate([x[1] for x in inference_batch], sampling_params)
                for (it, _), out in zip(inference_batch, outputs):
                    it["caption"] = out.outputs[0].text
                    out_file = OUT_DIR / f"{it['idx']}.json"
                    out_file.write_text(json.dumps(it, ensure_ascii=False, indent=2), encoding='utf-8')
                
                pbar.update(len(inference_batch))
            
            elif processed >= len(batch) and cache:
                outputs = llm.generate([x[1] for x in cache], sampling_params)
                for (it, _), out in zip(cache, outputs):
                    it["caption"] = out.outputs[0].text
                    out_file = OUT_DIR / f"{it['idx']}.json"
                    out_file.write_text(json.dumps(it, ensure_ascii=False, indent=2), encoding='utf-8')
                
                pbar.update(len(cache))
                cache.clear()
        
        pbar.close()
        
        torch.cuda.synchronize()
        torch.cuda.empty_cache()


    print('[Main] Sending STOP signals to workers...')
    for _ in range(NUM_WORKERS):
        task_q.put('STOP')
    

    finish_cnt = 0
    while finish_cnt < NUM_WORKERS:
        item, _ = result_q.get()
        if item is None:
            finish_cnt += 1
    
    for p in procs:
        p.join()
    
    print('[Main] All workers stopped. All finished.')

if __name__ == "__main__":
    main()
