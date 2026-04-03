python prepare_conversation.py \
    --input_manifest /apdcephfs_cq12/share_302080740/data/asr_train_data/manifests/chinese/open_source/aishell1/aishell1_cuts.jsonl.gz \
    --output_manifest data/train/aishell1_cuts_conversation.jsonl.gz \
    --tokenizer /apdcephfs_cq12/share_302080740/model/Qwen2-7B-Instruct \
    --instruction "Please transcribe speech." \