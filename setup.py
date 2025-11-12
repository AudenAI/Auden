from pathlib import Path

from setuptools import find_packages, setup

README = Path(__file__).parent / "README.md"
long_description = README.read_text(encoding="utf-8") if README.exists() else ""

setup(
    name="auden",
    version="0.1.0",
    description="Auden: Audio & Multimodal research toolbox (ASR, ST, CLAP, Speech-LLM, MLLM, TTA, Zipformer, etc.)",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="AudenAI",
    license="Apache-2.0",
    python_requires=">=3.10",
    package_dir={"": "src"},
    packages=find_packages(where="src", exclude=("examples*", "scripts*", "assets*")),
    include_package_data=True,
    install_requires=[
        # Core runtime
        "torch>=2.1",
        "torchaudio>=2.1",
        "hydra-core>=1.3",
        "omegaconf>=2.3",
        "transformers>=4.40",
        "sentencepiece>=0.1.99",
        "safetensors>=0.4",
        "numpy>=1.22,<2.0",
        "PyYAML>=6.0",
        "huggingface_hub>=0.20",
        "six>=1.16.0",
        "tensorboard>=2.12",
        "lhotse>=1.20",
        "kaldialign>=0.7",
        "tqdm>=4.65",
        "soundfile>=0.12",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
    ],
    url="https://github.com/AudenAI/Auden",
    project_urls={
        "Source": "https://github.com/AudenAI/Auden",
        "Issues": "https://github.com/AudenAI/Auden/issues",
    },
)
