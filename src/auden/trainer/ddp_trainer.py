import copy
import logging
from abc import ABC, abstractmethod
from pathlib import Path

import torch
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter

# trainer checkpointing (for resuming training - includes model + optimizer + scheduler + scaler + progress)
from ..utils.checkpoint import (
    load_trainer_checkpoint,
    remove_trainer_checkpoints,
    save_trainer_checkpoint,
    update_averaged_model,
)

# Metrics and tracking
from ..utils.metric_tracker import MetricsTracker


class BaseTrainer(ABC):
    """
    Abstract base trainer class for DDP (DistributedDataParallel) training.

    Provides core training infrastructure including:
    - Multi-GPU distributed training with PyTorch DDP
    - Mixed precision training with automatic loss scaling
    - Model averaging and checkpointing
    - Validation and logging with TensorBoard support
    - Training loop management and orchestration

    This is an abstract base class that must be extended by task-specific trainers.
    Subclasses must implement the abstract methods for their specific optimization
    and forward pass logic. Additionally, subclasses may override other methods
    like __init__ and validate for task-specific customization.

    Args:
        cfg: Training configuration object containing all hyperparameters
        model: PyTorch model to train
        data_module: Data module containing train/validation dataloaders
        rank: Global rank of current process (default: 0)
        local_rank: Local rank within current node (default: 0)
        world_size: Total number of processes (default: 1)

    Example:
        ```python
        class MyTrainer(BaseTrainer):
            def build_optimizer(self, model):
                return torch.optim.Adam(model.parameters(), lr=1e-4)

            def build_scheduler(self, optimizer):
                return torch.optim.lr_scheduler.StepLR(optimizer, step_size=10)

            def _forward_one_batch(self, batch, is_training=True):
                outputs = self.model(batch["inputs"])
                loss = F.cross_entropy(outputs, batch["targets"])
                metrics = {"loss": loss.item()}
                return loss, metrics

        trainer = MyTrainer(cfg, model, data_module, rank, local_rank, world_size)
        trainer.run()
        ```
    """

    def __init__(self, cfg, model, data_module, rank=0, local_rank=0, world_size=1):
        """
        Initialize the BaseTrainer with configuration, model, and data module.

        Args:
            cfg: Training configuration object containing all hyperparameters
            model: PyTorch model to train
            data_module: Data module containing train/validation dataloaders
            rank: Global rank of current process (default: 0)
            local_rank: Local rank within current node (default: 0)
            world_size: Total number of processes (default: 1)

        Note:
            Task-specific trainers may override this method to add custom initialization
            logic, additional setup, or task-specific configuration validation.
            The base implementation provides core infrastructure setup that most trainers
            will need, but subclasses can extend it as needed.
        """
        self.cfg = cfg
        self.exp_dir = cfg.exp_dir
        self.rank = rank
        self.local_rank = local_rank
        self.world_size = world_size
        self.device = torch.device("cuda", local_rank)
        self.use_fp16 = cfg.trainer.use_fp16
        self.global_step = cfg.trainer.start_batch
        self.tb_writer = None
        if self.rank == 0 and cfg.trainer.tensorboard:
            self.tb_writer = SummaryWriter(log_dir=f"{self.exp_dir}/tensorboard")

        # data module
        self.data_module = data_module

        # setup model
        self.model, self.model_avg = self.setup_model(model)

        # optimizer and scheduler
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.use_fp16)
        self.optimizer = self.build_optimizer(self.model)
        self.scheduler = self.build_scheduler(self.optimizer)

    def setup_model(self, model):
        """
        Setup model for training with distributed data parallel support.

        Initializes the model for training by moving it to the appropriate device,
        wrapping it with DistributedDataParallel if multiple GPUs are available,
        and creating a model averaging instance on rank 0 for EMA.

        Args:
            model (torch.nn.Module): The model to initialize for training

        Returns:
            tuple: A tuple containing:
                - Initialized model (potentially wrapped with DDP)
                - Model averaging instance (on CPU, rank 0 only) or None

        Note:
            - Model is moved to the device specified by local_rank
            - DDP wrapping is applied when world_size > 1
            - Model averaging is only created on rank 0 to save memory
            - Parameter statistics are logged for monitoring
        """
        if self.rank == 0:
            model_avg = copy.deepcopy(model).to(torch.float64).to("cpu")
        else:
            model_avg = None

        model = model.to(self.device)
        if self.world_size > 1:
            model = DDP(
                model,
                device_ids=[self.local_rank],
                find_unused_parameters=self.cfg.trainer.get(
                    "find_unused_parameters", True
                ),
            )

        num_param = sum(p.numel() for p in model.parameters()) / 1e6
        num_trainable_param = (
            sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6
        )
        logging.info(f"Number of model parameters: {num_param:.2f}M")
        logging.info(
            f"Number of trainable model parameters: {num_trainable_param:.2f}M"
        )
        return model, model_avg

    def build_optimizer(self, model):
        """
        Build optimizer based on configuration, with sensible defaults.

        Default behavior:
        - Supports "adamw" (default), "adam", and "scaled_adam".
        - Reads hyperparameters from ``cfg.trainer.optimizer`` with safe fallbacks.
        - Uses trainable parameters by default; subclasses may override
          ``get_parameter_groups()`` for custom grouping/freezing.

        Subclasses can override this method entirely for task-specific logic.
        """

        def _get(opt_cfg, key, default=None):
            # Works with both dict-like and attribute-like config objects
            try:
                v = getattr(opt_cfg, key)
            except Exception:
                v = None
            if v is None:
                try:
                    return opt_cfg.get(key, default)
                except Exception:
                    return default
            return v

        opt_cfg = getattr(self.cfg.trainer, "optimizer", {})
        name = (_get(opt_cfg, "type") or _get(opt_cfg, "name") or "adamw").lower()
        lr = float(_get(opt_cfg, "lr", 1e-3))
        weight_decay = float(_get(opt_cfg, "weight_decay", 0.0))
        betas = tuple(_get(opt_cfg, "betas", (0.9, 0.999)))
        eps = float(_get(opt_cfg, "eps", 1e-8))
        fused = bool(_get(opt_cfg, "fused", False))

        # Allow subclasses to supply custom parameter groups
        param_groups = self.get_parameter_groups(model, lr)

        if name == "adamw":
            adamw_kwargs = {
                "lr": lr,
                "weight_decay": weight_decay,
                "betas": betas,
                "eps": eps,
            }
            if fused:
                try:
                    return torch.optim.AdamW(param_groups, fused=True, **adamw_kwargs)
                except TypeError:
                    pass
            return torch.optim.AdamW(param_groups, **adamw_kwargs)

        if name == "adam":
            adam_kwargs = {
                "lr": lr,
                "betas": betas,
                "eps": eps,
                "weight_decay": weight_decay,
            }
            if fused:
                try:
                    return torch.optim.Adam(param_groups, fused=True, **adam_kwargs)
                except TypeError:
                    pass
            return torch.optim.Adam(param_groups, **adam_kwargs)

        if name == "scaled_adam":
            # Lazy import to keep base import light
            from ..optim import ScaledAdam, get_parameter_groups_with_lrs

            clipping_scale = float(_get(opt_cfg, "clipping_scale", 2.0))
            scalar_lr_scale = float(_get(opt_cfg, "scalar_lr_scale", 0.1))
            # Build named parameter groups with per-module lr scaling, supports freezing
            groups = get_parameter_groups_with_lrs(
                model,
                lr=lr,
                include_names=True,
            )
            return ScaledAdam(
                groups,
                lr=lr,
                clipping_scale=clipping_scale,
            )
        raise ValueError(
            f"Unsupported optimizer: {name}. Supported: adamw, adam, scaled_adam"
        )

    def get_parameter_groups(self, model, lr):
        """
        Return parameter groups for optimizers.

        Default: a flat list of trainable parameters. Subclasses may override to
        return a list of dicts with custom group options (e.g., different LRs or
        weight decays) or to freeze modules.
        """
        return [p for p in model.parameters() if p.requires_grad]

    def build_scheduler(self, optimizer):
        """
        Build learning rate scheduler based on configuration, with sensible defaults.

        Supports:
          - "eden" (Eden/Eden2/Eden3 from auden.optim)
          - "step", "multistep", "cosine", "cosine_restart", "exp" (torch schedulers via adapter)

        Subclasses can override for task-specific schedulers.
        """

        def _get(sch_cfg, key, default=None):
            try:
                v = getattr(sch_cfg, key)
            except Exception:
                v = None
            if v is None:
                try:
                    return sch_cfg.get(key, default)
                except Exception:
                    return default
            return v

        # Adapter to provide step_epoch/step_batch and get_last_lr for torch schedulers
        class _TorchSchedulerAdapter:
            def __init__(self, torch_scheduler, update_on="epoch"):
                self._sch = torch_scheduler
                self._last_lr = [g["lr"] for g in self._sch.optimizer.param_groups]
                self._update_on = update_on

            def get_last_lr(self):
                return self._last_lr

            def step_batch(self, batch: int | None = None):
                if self._update_on == "batch":
                    self._sch.step()
                    self._last_lr = [g["lr"] for g in self._sch.optimizer.param_groups]

            def step_epoch(self, epoch: int | None = None):
                if self._update_on == "epoch":
                    self._sch.step()
                    self._last_lr = [g["lr"] for g in self._sch.optimizer.param_groups]

        sch_cfg = getattr(self.cfg.trainer, "scheduler", {})
        name = (_get(sch_cfg, "type") or _get(sch_cfg, "name") or "eden").lower()

        # Eden family (native auden LRScheduler)
        if name in {"eden", "eden2", "eden3"}:
            from ..optim import Eden, Eden2, Eden3

            warmup_batches = _get(sch_cfg, "warmup_batches", 500.0)
            use_inf = bool(getattr(self.cfg.data, "use_infinite_dataset", False))

            if name == "eden":
                lr_batches = _get(sch_cfg, "lr_batches", 5000)
                lr_epochs = _get(sch_cfg, "lr_epochs", 6)
                if use_inf:
                    steps_per_epoch = _get(sch_cfg, "lr_steps_per_epoch", 0)
                    if steps_per_epoch and steps_per_epoch > 0:
                        return Eden3(
                            optimizer,
                            lr_batches,
                            lr_epochs,
                            steps_per_epoch,
                            warmup_batches,
                        )
                    else:
                        return Eden2(optimizer, lr_batches, warmup_batches)
                else:
                    return Eden(optimizer, lr_batches, lr_epochs, warmup_batches)

            if name == "eden2":
                lr_batches = _get(sch_cfg, "lr_batches", 5000)
                return Eden2(optimizer, lr_batches, warmup_batches)

            if name == "eden3":
                lr_batches = _get(sch_cfg, "lr_batches", 5000)
                lr_epochs = _get(sch_cfg, "lr_epochs", 6)
                steps_per_epoch = _get(sch_cfg, "lr_steps_per_epoch", 100000)
                return Eden3(
                    optimizer, lr_batches, lr_epochs, steps_per_epoch, warmup_batches
                )

        # Common torch schedulers (wrapped)
        import torch.optim.lr_scheduler as lr_sch

        if name == "step":
            step_size = int(_get(sch_cfg, "step_size", 1))
            gamma = float(_get(sch_cfg, "gamma", 0.1))
            return _TorchSchedulerAdapter(
                lr_sch.StepLR(optimizer, step_size=step_size, gamma=gamma),
                update_on="epoch",
            )

        if name == "multistep":
            milestones = list(_get(sch_cfg, "milestones", []))
            gamma = float(_get(sch_cfg, "gamma", 0.1))
            return _TorchSchedulerAdapter(
                lr_sch.MultiStepLR(optimizer, milestones=milestones, gamma=gamma),
                update_on="epoch",
            )

        if name == "cosine":
            # Epoch-based cosine annealing
            t_max = int(
                _get(sch_cfg, "t_max", getattr(self.cfg.trainer, "num_epochs", 100))
            )
            eta_min = float(_get(sch_cfg, "eta_min", 0.0))
            return _TorchSchedulerAdapter(
                lr_sch.CosineAnnealingLR(optimizer, T_max=t_max, eta_min=eta_min),
                update_on="epoch",
            )

        if name in {"cosine_restart", "cosine_warm_restarts"}:
            t_0 = int(_get(sch_cfg, "t_0", 10))
            t_mult = int(_get(sch_cfg, "t_mult", 1))
            eta_min = float(_get(sch_cfg, "eta_min", 0.0))
            return _TorchSchedulerAdapter(
                lr_sch.CosineAnnealingWarmRestarts(
                    optimizer, T_0=t_0, T_mult=t_mult, eta_min=eta_min
                ),
                update_on="epoch",
            )

        if name in {"exp", "exponential"}:
            gamma = float(_get(sch_cfg, "gamma", 0.95))
            return _TorchSchedulerAdapter(
                lr_sch.ExponentialLR(optimizer, gamma=gamma), update_on="epoch"
            )

        raise ValueError(
            f"Unsupported scheduler: {name}. Supported: eden, eden2, eden3, step, multistep, cosine, cosine_restart, exp"
        )

    @abstractmethod
    def _forward_one_batch(self, batch, is_training=True):
        """
        Perform forward pass for one batch of data.

        This is an abstract method that must be implemented by task-specific trainers.
        Each trainer should implement its own forward pass logic based on the
        specific requirements of the task.

        Args:
            batch: Training batch data containing inputs and targets
            is_training: Whether the model is in training mode

        Returns:
            tuple: A tuple containing:
                - loss (Tensor): The scalar loss value
                - metrics: Dictionary or MetricsTracker object containing logging info

        Note:
            Task-specific trainers should implement their own forward pass logic here,
            including model forward pass, loss calculation, and metrics computation.
        """
        pass

    def resume_training_from_checkpoint(self):
        """
        Resume training from a checkpoint if specified in configuration.

        Loads model state, optimizer state, scheduler state, and training progress
        from either step-based or epoch-based checkpoints. This method supports
        resuming from both step-based checkpoints (checkpoint-{step}.pt) and
        epoch-based checkpoints (epoch-{epoch}.pt).

        Checkpoint Resolution:
            - If start_batch > 0: looks for checkpoint-{start_batch}.pt
            - If start_epoch > 1: looks for epoch-{start_epoch-1}.pt
            - If neither is specified, no checkpoint is loaded

        Raises:
            FileNotFoundError: If specified checkpoint file doesn't exist

        Note:
            The global_step is updated from the checkpoint's batch_idx_train value.
            All training state (model, optimizer, scheduler, scaler) is restored.
        """
        resume_ckpt = None
        if self.cfg.trainer.start_batch > 0:
            resume_ckpt = (
                Path(self.exp_dir) / f"checkpoint-{self.cfg.trainer.start_batch}.pt"
            )
        elif self.cfg.trainer.start_epoch > 1:
            resume_ckpt = (
                Path(self.exp_dir) / f"epoch-{self.cfg.trainer.start_epoch - 1}.pt"
            )

        if resume_ckpt and resume_ckpt.is_file():
            logging.info(f"Resuming training from: {resume_ckpt}")
            checkpoints = load_trainer_checkpoint(
                resume_ckpt,
                model=self.model,
                model_avg=self.model_avg,
                optimizer=self.optimizer,
                scheduler=self.scheduler,
                scaler=self.scaler,
            )
            self.global_step = checkpoints.get("batch_idx_train", self.global_step)
        elif resume_ckpt:
            raise FileNotFoundError(f"Checkpoint file not found: {resume_ckpt}")

    def train_one_epoch(self, epoch: int):
        """
        Train the model for one complete epoch.

        Handles the full training loop including forward/backward passes,
        periodic validation, checkpointing, model averaging, and logging.
        Processes all batches in the training dataloader and manages training
        state updates throughout the epoch.

        Args:
            epoch: Current epoch number (1-indexed)

        Note:
            The training loop includes:
            - Batch processing with mixed precision support
            - Periodic model averaging (if enabled)
            - Periodic validation and checkpointing
            - Gradient scaling monitoring for FP16 training
            - Training status logging
            - Memory cleanup at the end of the epoch
        """
        self.model.train()
        metrics_tracker = MetricsTracker()

        for batch_idx, batch in enumerate(self.data_module.train_dl):
            # Optional: sync ScheduledFloat batch_count to global_step
            if batch_idx % 10 == 0 and self.global_step < 100000:
                self._maybe_update_batch_count()
            if self.cfg.data.use_infinite_dataset:
                batch_idx = self.global_step

            self.global_step += 1
            batch_size = batch["inputs"].size(0)

            loss, batch_metrics = self._forward_backward_optimize(batch)

            # Track metrics
            metrics_tracker.update(batch_metrics, self.cfg.trainer.reset_interval)

            # Periodically update model avg
            self._maybe_update_model_average()

            # Periodically evaluate and save
            self._maybe_validate_and_save(epoch)

            # Monitor and adjust AMP grad scale
            self._maybe_rescale_grad_amp(batch_idx)

            # Periodic logging
            self._maybe_log_training_status(
                epoch, batch_idx, batch_size, batch_metrics, metrics_tracker
            )

            if self.global_step > self.cfg.trainer.num_steps:
                break

        torch.cuda.empty_cache()

    def validate(self, epoch: int):
        """
        Runs validation on one or more validation sets.

        Evaluates the model on validation data and logs metrics. Supports multiple
        validation datasets through self.data_module.valid_dls. In distributed training,
        metrics are reduced across all processes before logging.

        Args:
            epoch: Current epoch number for logging purposes

        Note:
            Uses self.data_module.valid_dls (can be a list of loaders).
            Only logs on rank 0 to avoid duplicate logging in distributed training.
            The model is set to evaluation mode during validation and back to training
            mode after completion.

            Task-specific trainers may override this method to implement custom
            validation logic, task-specific metrics computation, or specialized
            validation workflows (e.g., beam search for ASR, different evaluation
            protocols for different tasks).
        """
        self.model.eval()
        with torch.no_grad():
            for valid_name, valid_dl in zip(
                self.data_module.valid_names, self.data_module.valid_dls
            ):
                total_metrics = MetricsTracker()
                for batch in valid_dl:
                    loss, batch_metrics = self._forward_one_batch(
                        batch=batch, is_training=False
                    )
                    assert not loss.requires_grad
                    total_metrics.update(batch_metrics)

                # DDP reduce
                if self.world_size > 1:
                    total_metrics.reduce(device=loss.device)

                # Logging
                if self.rank == 0:
                    logging.info(
                        f"Epoch {epoch}, global step {self.global_step}, validation: {total_metrics}"
                    )

                    if self.tb_writer is not None:
                        total_metrics.write_summary(
                            self.tb_writer,
                            f"train/valid_{valid_name}_",
                            self.global_step,
                        )

        self.model.train()

    def run(self):
        """
        Main entry point to run the complete training process.

        Orchestrates the full training workflow including checkpoint resumption,
        epoch-based training, validation, and final checkpoint saving. This method
        manages the entire training lifecycle from initialization to completion.

        Training Process:
            1. Resume from checkpoint if specified in configuration
            2. Iterate through epochs from start_epoch to num_epochs
            3. For each epoch:
               - Advance epoch-based scheduler
               - Set deterministic behavior for data loading
               - Train for one complete epoch
               - Run validation
               - Save epoch checkpoint
            4. Stop when num_steps is reached or all epochs completed

        Note:
            The training can be stopped early if global_step exceeds num_steps.
            Epoch-based checkpoints are saved after each epoch completion.
        """
        # resume training
        self.resume_training_from_checkpoint()

        num_epochs = self.cfg.trainer.num_epochs
        start_epoch = self.cfg.trainer.start_epoch
        num_steps = self.cfg.trainer.num_steps

        for epoch in range(start_epoch, num_epochs + 1):
            # Advance epoch-based scheduler (optional)
            self.scheduler.step_epoch(epoch - 1)

            # Ensure deterministic behavior across workers (e.g., for shuffling)
            if hasattr(self.data_module.train_dl, "sampler") and hasattr(
                self.data_module.train_dl.sampler, "set_epoch"
            ):
                self.data_module.train_dl.sampler.set_epoch(epoch - 1)

            # Log epoch marker
            if self.tb_writer and self.rank == 0:
                self.tb_writer.add_scalar("train/epoch", epoch, self.global_step)

            # Train for one epoch
            self.train_one_epoch(epoch)

            # Full validation pass (optional — if not handled inside train loop)
            self.validate(epoch)

            # Save full checkpoint (epoch-based)
            self._maybe_save_epoch_checkpoint(epoch)

            if self.global_step > num_steps:
                logging.info(
                    f"Reached global_step={self.global_step} > num_steps={num_steps}. Stopping training."
                )
                break

    def _maybe_update_batch_count(self):
        """
        Optionally update modules' `batch_count` to the current global_step.

        This is used by architectures that rely on a global step counter to evaluate
        ScheduledFloat values in training. It is gated by `trainer.set_batch_count` and
        called periodically in the training loop.
        """
        model = (
            self.model.module
            if isinstance(self.model, torch.nn.parallel.DistributedDataParallel)
            else self.model
        )
        for m in model.modules():
            if hasattr(m, "set_batch_count"):
                if m.model_batch_count < 100000:
                    m.model_batch_count.add_(10)
                    m.set_batch_count()

    def _forward_backward_optimize(self, batch):
        """
        Performs a forward pass, backward pass, and optimizer step with mixed precision.

        This method handles the core training step including:
        1. Forward pass with automatic mixed precision (AMP) if enabled
        2. Backward pass with gradient scaling for FP16 training
        3. Optimizer step and learning rate scheduling
        4. Gradient zeroing for the next iteration

        Args:
            batch (dict): A batch of training data containing inputs and targets

        Returns:
            tuple: A tuple containing:
                - loss (Tensor): The scalar loss value before scaling
                - batch_metrics: Dictionary or MetricsTracker object containing logging info

        Note:
            This method uses automatic mixed precision when self.use_fp16 is True.
            The gradient scaler is used to prevent gradient underflow in FP16 training.
        """
        with torch.amp.autocast("cuda", enabled=self.use_fp16):
            loss, batch_metrics = self._forward_one_batch(batch=batch, is_training=True)

        # Backprop and optimization step
        self.scaler.scale(loss).backward()
        self.scheduler.step_batch(self.global_step)
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.optimizer.zero_grad()

        return loss, batch_metrics

    def _maybe_update_model_average(self):
        """
        Updates the averaged model on rank 0 using uniform averaging.

        This method maintains a uniform average of model parameters on the CPU,
        which can improve model stability and generalization. The averaged
        model is updated at regular intervals during training.

        Requires:
            - self.model_avg (on CPU): The averaged model to update
            - self.model (on GPU, optionally wrapped in DDP): Current training model
            - self.cfg.trainer.average_period: Interval for updating the average
            - self.global_step: Current training step for timing

        Note:
            Only runs on rank 0 to avoid duplicate averaging in distributed training.
            The averaged model is typically used for inference or final model selection.
        """
        if (
            self.rank != 0
            or not self.cfg.trainer.get("use_averaged_model", False)
            or self.global_step == 0
            or self.global_step % self.cfg.trainer.average_period != 0
        ):
            return
        update_averaged_model(
            average_period=self.cfg.trainer.average_period,
            batch_idx_train=self.global_step,
            model_cur=self.model,
            model_avg=self.model_avg,
        )

    def _maybe_validate_and_save(self, epoch: int):
        """
        Periodically run validation and save a step-based checkpoint.

        Args:
            epoch: Current epoch number for validation logging

        Note:
            This should be called during training at intervals specified by
            cfg.trainer.valid_interval. Only saves checkpoints when both validation
            and save conditions are met (based on cfg.trainer.save_every_n).
            Only executes on rank 0 to avoid duplicate operations in distributed training.
        """
        if self.global_step == 0:
            return

        if self.global_step % self.cfg.trainer.valid_interval != 0:
            return

        self.validate(epoch)

        save_every = self.cfg.trainer.save_every_n * self.cfg.trainer.valid_interval
        if self.global_step % save_every != 0:
            return

        if self.rank == 0:
            ckpt_path = Path(self.exp_dir) / f"checkpoint-{self.global_step}.pt"
            save_trainer_checkpoint(
                filename=ckpt_path,
                model=self.model,
                model_avg=self.model_avg,
                batch_idx_train=self.global_step,
                optimizer=self.optimizer,
                scheduler=self.scheduler,
                scaler=self.scaler,
                sampler=self.data_module.train_dl.sampler,
                rank=self.rank,
            )
            logging.info(f"Saving trainer checkpoint to: {ckpt_path}")
            remove_trainer_checkpoints(
                out_dir=self.exp_dir,
                topk=self.cfg.trainer.keep_last_k,
                rank=self.rank,
            )

    def _maybe_rescale_grad_amp(self, batch_idx: int):
        """
        Monitors and adjusts the AMP gradient scaler if it's too small.

        Helps recover from instability early in training by proactively increasing
        the gradient scale when it becomes too small, which can happen during
        mixed precision training.

        Args:
            batch_idx: Current batch index for timing the rescaling checks

        Note:
            This only runs if use_fp16 is True and checks every 100 batches.
            If the scale becomes extremely small (< 1e-5), training will be terminated
            as this indicates severe numerical instability.
        """
        if not self.use_fp16 or batch_idx % 100 != 0:
            return

        cur_scale = self.scaler.get_scale()

        # Proactively increase the scale if it's growing too slowly
        if cur_scale < 8.0 or (cur_scale < 32.0 and batch_idx % 400 == 0):
            self.scaler.update(cur_scale * 2.0)

        # Warn or crash if scale is extremely small
        if cur_scale < 0.01:
            logging.warning(f"Grad scale is small: {cur_scale}")
        if cur_scale < 1.0e-5:
            raise RuntimeError(f"grad_scale is too small, exiting: {cur_scale}")

    def _maybe_log_training_status(
        self,
        epoch: int,
        batch_idx: int,
        batch_size: int,
        batch_metrics,
        total_metrics,
    ):
        """
        Logs training status and writes TensorBoard scalars at configured intervals.

        Args:
            epoch: Current epoch number (1-indexed)
            batch_idx: Current batch index within the epoch
            batch_size: Size of the current batch
            batch_metrics: Metrics for the current batch
            total_metrics: Accumulated metrics across batches

        Note:
            Only executes on rank 0 to avoid duplicate logging in distributed training.
            Logs are written at intervals specified by cfg.trainer.log_interval.
        """
        if self.rank != 0 or batch_idx % self.cfg.trainer.log_interval != 0:
            return

        cur_lr = max(self.scheduler.get_last_lr())
        cur_grad_scale = self.scaler.get_scale() if self.use_fp16 else 1.0

        logging.info(
            f"Epoch {epoch}, "
            f"batch {batch_idx}, info[{batch_metrics}], "
            f"tot_info[{total_metrics}], batch size: {batch_size}, "
            f"lr: {cur_lr:.2e}, "
            + (f"grad_scale: {cur_grad_scale}" if self.use_fp16 else "")
        )

        if self.tb_writer is not None:
            self.tb_writer.add_scalar("train/learning_rate", cur_lr, self.global_step)
            if self.use_fp16:
                self.tb_writer.add_scalar(
                    "train/grad_scale", cur_grad_scale, self.global_step
                )

            batch_metrics.write_summary(
                self.tb_writer, "train/current_", self.global_step
            )
            total_metrics.write_summary(self.tb_writer, "train/tot_", self.global_step)

    def _maybe_save_epoch_checkpoint(self, epoch: int):
        """
        Saves an epoch-level checkpoint (outside step-based logic).

        Args:
            epoch: Current epoch number to save checkpoint for

        Note:
            Only executes on rank 0 to avoid duplicate checkpoint saving in distributed training.
            Saves the checkpoint with filename format: epoch-{epoch}.pt
        """
        if self.rank != 0:
            return

        filename = Path(self.exp_dir) / f"epoch-{epoch}.pt"
        logging.info(f"Saving epoch checkpoint to: {filename}")

        save_trainer_checkpoint(
            filename=filename,
            batch_idx_train=self.global_step,
            model=self.model,
            model_avg=self.model_avg,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            sampler=self.data_module.train_dl.sampler,
            scaler=self.scaler,
            rank=self.rank,
        )
