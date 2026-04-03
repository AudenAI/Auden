import logging
import os

import hydra
import torch
import torch.distributed as dist
from lalm_core import LALMDataModule, LALMTrainer
from lalm_core.model import LALMConfig, LALMForConditionalGeneration, LALMProcessor
from lhotse.utils import fix_random_seed
from omegaconf import DictConfig, OmegaConf


def _resolve_dtype(name: str | None) -> torch.dtype:
    if name == "bf16":
        return torch.bfloat16
    if name == "fp16":
        return torch.float16
    return torch.float32


def _freeze_modules(model: torch.nn.Module, names: list[str]) -> None:
    for name in names:
        module = model
        for part in name.split("."):
            module = getattr(module, part, None)
            if module is None:
                logging.warning(f"[train] Module not found, skip freeze: {name}")
                break
        if module is not None:
            for p in module.parameters():
                p.requires_grad = False


@hydra.main(version_base=None, config_path="configs", config_name="train")
def main(cfg: DictConfig):
    logging.info("\n" + OmegaConf.to_yaml(cfg))

    # 1) Fix random seed
    if "seed" in cfg:
        fix_random_seed(cfg.seed)

    # 2) Gather torchrun environment variables
    rank = int(os.environ.get("RANK", 0))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    if world_size > 1:
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl", init_method="env://")

    if cfg.get("exp_dir"):
        os.makedirs(cfg.exp_dir, exist_ok=True)

    dtype = _resolve_dtype(cfg.trainer.get("mixed_precision", None))
    start_batch = int(cfg.trainer.get("start_batch", 0))
    start_epoch = int(cfg.trainer.get("start_epoch", 0))
    hf_dir = os.path.join(cfg.exp_dir, "hf") if cfg.get("exp_dir") else None

    # 3) Match BaseTrainer semantics:
    #    - fresh start when start_batch == 0 and start_epoch <= 1
    #    - resume from step checkpoint when start_batch > 0
    #    - resume from epoch checkpoint when start_epoch > 1
    fresh_start = start_batch == 0 and start_epoch == 1
    if not fresh_start:
        if not hf_dir or not os.path.isdir(hf_dir):
            raise FileNotFoundError(
                f"[train] Resume requested (start_batch={start_batch}, "
                f"start_epoch={start_epoch}) but hf_dir not found: {hf_dir}"
            )
        logging.info(
            f"[train] Resume mode: init model from config in {hf_dir}; "
            "trainer will restore checkpoint weights."
        )
        model_cfg = LALMConfig.from_pretrained(hf_dir)
        from transformers.modeling_utils import no_init_weights

        with no_init_weights():
            model = LALMForConditionalGeneration(model_cfg)
        model = model.to(dtype)
        processor = LALMProcessor.from_pretrained(hf_dir)
    else:
        model_dir = cfg.model.pretrained_model
        logging.info(f"[train] Fresh start: loading model from {model_dir}")
        model = LALMForConditionalGeneration.from_pretrained(
            model_dir, torch_dtype=dtype
        )
        processor = LALMProcessor.from_pretrained(model_dir)
        if rank == 0:
            os.makedirs(hf_dir, exist_ok=True)
            model.config.save_pretrained(hf_dir)
            processor.save_pretrained(hf_dir)
            logging.info(f"[train] Saved config + processor to {hf_dir}")

    # 4) Freeze requested modules
    frozen = list(cfg.trainer.get("frozen_modules", []) or [])
    if frozen:
        _freeze_modules(model, frozen)

    # 5) Initialize data module
    data_module = LALMDataModule(cfg.data, processor)

    # 6) Create trainer and run
    trainer = LALMTrainer(
        cfg,
        model,
        data_module,
        rank=rank,
        local_rank=local_rank,
        world_size=world_size,
    )
    trainer.run()

    if world_size > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
