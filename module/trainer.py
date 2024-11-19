import contextlib
import copy
import functools
import glob
import importlib.metadata
import inspect
import math
import os
import random
import re
import shutil
import sys
import time
import warnings
from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Union
from torch.utils.data import DataLoader, Dataset

from transformers import (
    Trainer, 
    logging,
    __version__,
)

from transformers.trainer_utils import (
    has_length
)
import math
import os
import shutil
import sys
import time
import transformers

from transformers.integrations import (
    get_reporting_integration_callbacks,
    hp_params,
)

# isort: on

import huggingface_hub.utils as hf_hub_utils
import torch
import torch.distributed as dist
from packaging import version
from torch import nn
from torch.utils.data import RandomSampler

from transformers.configuration_utils import PretrainedConfig
from transformers.data.data_collator import DataCollator, DataCollatorWithPadding, default_data_collator
from transformers.debug_utils import DebugOption, DebugUnderflowOverflow
from transformers.integrations.deepspeed import deepspeed_init, deepspeed_load_checkpoint, is_deepspeed_available
from transformers.modelcard import TrainingSummary
from transformers.modeling_utils import PreTrainedModel, load_sharded_checkpoint, unwrap_model
from transformers.models.auto.modeling_auto import MODEL_FOR_CAUSAL_LM_MAPPING_NAMES, MODEL_MAPPING_NAMES
from transformers.optimization import Adafactor, get_scheduler
from transformers.pytorch_utils import ALL_LAYERNORM_LAYERS
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from transformers.trainer_callback import (
    CallbackHandler,
    DefaultFlowCallback,
    PrinterCallback,
    ProgressCallback,
    TrainerCallback,
    TrainerControl,
    TrainerState,
)
from transformers.trainer_pt_utils import (
    DistributedTensorGatherer,
    IterableDatasetShard,
    LabelSmoother,
    LengthGroupedSampler,
    SequentialDistributedSampler,
    distributed_broadcast_scalars,
    distributed_concat,
    find_batch_size,
    get_dataloader_sampler,
    get_model_param_count,
    get_module_class_from_name,
    get_parameter_names,
    nested_concat,
    nested_detach,
    nested_numpify,
    nested_xla_mesh_reduce,
    reissue_pt_warnings,
    remove_dummy_checkpoint,
)
from transformers.trainer_utils import (
    PREFIX_CHECKPOINT_DIR,
    BestRun,
    EvalLoopOutput,
    EvalPrediction,
    FSDPOption,
    HPSearchBackend,
    HubStrategy,
    IntervalStrategy,
    PredictionOutput,
    RemoveColumnsCollator,
    TrainerMemoryTracker,
    TrainOutput,
    default_compute_objective,
    denumpify_detensorize,
    enable_full_determinism,
    find_executable_batch_size,
    get_last_checkpoint,
    has_length,
    number_of_arguments,
    seed_worker,
    set_seed,
    speed_metrics,
)
from transformers.training_args import OptimizerNames, ParallelMode, TrainingArguments
from transformers.utils import (
    ADAPTER_CONFIG_NAME,
    ADAPTER_SAFE_WEIGHTS_NAME,
    ADAPTER_WEIGHTS_NAME,
    CONFIG_NAME,
    SAFE_WEIGHTS_INDEX_NAME,
    SAFE_WEIGHTS_NAME,
    WEIGHTS_INDEX_NAME,
    WEIGHTS_NAME,
    PushInProgress,
    can_return_loss,
    find_labels,
    is_accelerate_available,
    is_apex_available,
    is_bitsandbytes_available,
    is_datasets_available,
    is_in_notebook,
    is_ipex_available,
    is_peft_available,
    is_safetensors_available,
    is_sagemaker_dp_enabled,
    is_sagemaker_mp_enabled,
    is_torch_compile_available,
    is_torch_neuroncore_available,
    is_torch_tpu_available,
    logging,
    strtobool,
)
from transformers.utils.quantization_config import QuantizationMethod


DEFAULT_CALLBACKS = [DefaultFlowCallback]
DEFAULT_PROGRESS_CALLBACK = ProgressCallback

if is_in_notebook():
    from transformers.utils.notebook import NotebookProgressCallback

    DEFAULT_PROGRESS_CALLBACK = NotebookProgressCallback

if is_apex_available():
    from apex import amp

if is_datasets_available():
    import datasets

if is_torch_tpu_available(check_device=False):
    import torch_xla.core.xla_model as xm
    import torch_xla.debug.metrics as met


if is_sagemaker_mp_enabled():
    import smdistributed.modelparallel.torch as smp
    from smdistributed.modelparallel import __version__ as SMP_VERSION

    IS_SAGEMAKER_MP_POST_1_10 = version.parse(SMP_VERSION) >= version.parse("1.10")

    from transformers.trainer_pt_utils import smp_forward_backward, smp_forward_only, smp_gather, smp_nested_concat
else:
    IS_SAGEMAKER_MP_POST_1_10 = False


if is_safetensors_available():
    import safetensors.torch


if is_peft_available():
    from peft import PeftModel


if is_accelerate_available():
    from accelerate import Accelerator, skip_first_batches
    from accelerate import __version__ as accelerate_version
    from accelerate.utils import DistributedDataParallelKwargs, GradientAccumulationPlugin

    if version.parse(accelerate_version) > version.parse("0.20.3"):
        from accelerate.utils import (
            load_fsdp_model,
            load_fsdp_optimizer,
            save_fsdp_model,
            save_fsdp_optimizer,
        )

    if is_deepspeed_available():
        from accelerate.utils import DeepSpeedSchedulerWrapper


if TYPE_CHECKING:
    import optuna


logger = transformers.utils.logging.get_logger()


# Name of the files used for checkpointing
TRAINING_ARGS_NAME = "training_args.bin"
TRAINER_STATE_NAME = "trainer_state.json"
OPTIMIZER_NAME = "optimizer.pt"
OPTIMIZER_NAME_BIN = "optimizer.bin"
SCHEDULER_NAME = "scheduler.pt"
SCALER_NAME = "scaler.pt"

from utils.utils import ModelState
import copy

import math
import warnings
from functools import partial
from typing import Callable, Iterable, Optional, Tuple, Union

import torch
from torch import nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR, ReduceLROnPlateau

from transformers.trainer_utils import SchedulerType
from transformers.optimization import TYPE_TO_SCHEDULER_FUNCTION

from .sampler import DomainSampler


class CoTrainer(Trainer):
    """
    Trainer for align only.
    Alternatively switch a part of model to froze
    different part has its own logging/evaluation strategy, learning rate, scheduler
    """

    def create_optimizer(self):
        """
        Setup the optimizer.

        We provide a reasonable default that works well. If you want to use something else, you can pass a tuple in the
        Trainer's init through `optimizers`, or subclass and override this method in a subclass.
        """
        opt_model = self.model_wrapped if is_sagemaker_mp_enabled() else self.model

        if self.optimizer is None:
            decay_parameters = self.get_decay_parameter_names(opt_model)
            optimizer_grouped_parameters = [
                {
                    "params": [
                        p for n, p in opt_model.named_parameters() if (n in decay_parameters and p.requires_grad) and \
                                                                        'language_model' in n
                    ],
                    "lr": self.args.learning_rate,
                    "weight_decay": self.args.weight_decay,
                },
                {
                    "params": [
                        p for n, p in opt_model.named_parameters() if (n not in decay_parameters and p.requires_grad) and \
                                                                        'language_model' in n
                    ],
                    "lr": self.args.learning_rate,
                    "weight_decay": 0.0,
                },
                {
                    "params": [
                        p for n, p in opt_model.named_parameters() if (n in decay_parameters and p.requires_grad) and \
                                                                        ('language_model' not in n)
                    ],
                    "lr": self.model.config_dict['ctr_model']['learning_rate'],
                    "weight_decay": self.model.config_dict['ctr_model']['weight_decay'],
                },
                {
                    "params": [
                        p for n, p in opt_model.named_parameters() if (n not in decay_parameters and p.requires_grad) and \
                                                                        ('language_model' not in n)
                    ],
                    "lr": self.model.config_dict['ctr_model']['learning_rate'],
                    "weight_decay": 0.0,
                },
            ]

            optimizer_cls, optimizer_kwargs = Trainer.get_optimizer_cls_and_kwargs(self.args)

            self.optimizer = optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs)
            if optimizer_cls.__name__ == "Adam8bit":
                import bitsandbytes

                manager = bitsandbytes.optim.GlobalOptimManager.get_instance()

                skipped = 0
                for module in opt_model.modules():
                    if isinstance(module, nn.Embedding):
                        skipped += sum({p.data_ptr(): p.numel() for p in module.parameters()}.values())
                        logger.info(f"skipped {module}: {skipped/2**20}M params")
                        manager.register_module_override(module, "weight", {"optim_bits": 32})
                        logger.debug(f"bitsandbytes: will optimize {module} in fp32")
                logger.info(f"skipped: {skipped/2**20}M params")

        if is_sagemaker_mp_enabled():
            self.optimizer = smp.DistributedOptimizer(self.optimizer)

        return self.optimizer
    

    def _maybe_log_save_evaluate(self, tr_loss, model, trial, epoch, ignore_keys_for_eval):
        if self.control.should_log:
            if is_torch_tpu_available():
                xm.mark_step()

            logs: Dict[str, float] = {}

            # all_gather + mean() to get average loss over all processes
            tr_loss_scalar = self._nested_gather(tr_loss).mean().item()

            # reset tr_loss to zero
            tr_loss -= tr_loss

            logs["loss"] = round(tr_loss_scalar / (self.state.global_step - self._globalstep_last_logged), 4)
            logs["learning_rate"] = self._get_learning_rate()
            logs["ctr_learning_rate"] = self._get_ctr_learning_rate()   #

            self._total_loss_scalar += tr_loss_scalar
            self._globalstep_last_logged = self.state.global_step
            self.store_flos()

            self.log(logs)

        metrics = None
        if self.control.should_evaluate:
            if isinstance(self.eval_dataset, dict):
                metrics = {}
                for eval_dataset_name, eval_dataset in self.eval_dataset.items():
                    dataset_metrics = self.evaluate(
                        eval_dataset=eval_dataset,
                        ignore_keys=ignore_keys_for_eval,
                        metric_key_prefix=f"eval_{eval_dataset_name}",
                    )
                    metrics.update(dataset_metrics)
            else:
                metrics = self.evaluate(ignore_keys=ignore_keys_for_eval)
            self._report_to_hp_search(trial, self.state.global_step, metrics)

            # Run delayed LR scheduler now that metrics are populated
            if isinstance(self.lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                metric_to_check = self.args.metric_for_best_model
                if not metric_to_check.startswith("eval_"):
                    metric_to_check = f"eval_{metric_to_check}"
                self.lr_scheduler.step(metrics[metric_to_check])

        if self.control.should_save:
            self._save_checkpoint(model, trial, metrics=metrics)
            self.control = self.callback_handler.on_save(self.args, self.state, self.control)


    def _get_learning_rate(self):
        """
        language_model
        """
        if self.is_deepspeed_enabled:
            # with deepspeed's fp16 and dynamic loss scale enabled the optimizer/scheduler steps may
            # not run for the first few dozen steps while loss scale is too large, and thus during
            # that time `get_last_lr` will fail if called during that warm up stage, so work around it:
            try:
                last_lr = self.lr_scheduler.get_last_lr()[0]
            except AssertionError as e:
                if "need to call step" in str(e):
                    logger.warning("tried to get lr value before scheduler/optimizer started stepping, returning lr=0")
                    last_lr = 0
                else:
                    raise
        else:
            if isinstance(self.lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                last_lr = self.optimizer.param_groups[0]["lr"]  
            else:
                last_lr = self.lr_scheduler.get_last_lr()[0]
            if torch.is_tensor(last_lr):
                last_lr = last_lr.item()
        return last_lr
    

    def _get_ctr_learning_rate(self):
        """
        ctr_model
        """
        if self.is_deepspeed_enabled:
            raise
        else:
            if isinstance(self.lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                last_lr = self.optimizer.param_groups[-1]["lr"]  
            else:
                last_lr = self.lr_scheduler.get_last_lr()[-1]
            if torch.is_tensor(last_lr):
                last_lr = last_lr.item()
        return last_lr



class ContrastiveTrainer(Trainer):

    def _save(self, output_dir: Optional[str] = None, state_dict=None):
        # If we are executing this function, we are the process zero, so we don't check for that.
        output_dir = output_dir if output_dir is not None else self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        logger.info(f"Saving model checkpoint to {output_dir}")

        supported_classes = (PreTrainedModel,) if not is_peft_available() else (PreTrainedModel, PeftModel)
        # Save a trained model and configuration using `save_pretrained()`.
        # They can then be reloaded using `from_pretrained()`
        if not isinstance(self.model, supported_classes):
            if state_dict is None:
                state_dict = self.model.state_dict()

            if isinstance(unwrap_model(self.model), supported_classes):
                unwrap_model(self.model).save_pretrained(
                    output_dir, state_dict=state_dict, safe_serialization=self.args.save_safetensors
                )
            else:
                logger.info("Trainer.model is not a `PreTrainedModel`, only saving its state dict.")
                torch.save(state_dict, os.path.join(output_dir, WEIGHTS_NAME))
                self.save_language_model(output_dir)
        else:
            self.model.save_pretrained(
                output_dir, state_dict=state_dict, safe_serialization=self.args.save_safetensors
            )

        if self.tokenizer is not None:
            self.tokenizer.save_pretrained(output_dir)

        # Good practice: save your training arguments together with the trained model
        torch.save(self.args, os.path.join(output_dir, TRAINING_ARGS_NAME))

    def save_language_model(self, output_dir: Optional[str] = None, state_dict=None):
        # If we are executing this function, we are the process zero, so we don't check for that.
        output_dir = output_dir if output_dir is not None else self.args.output_dir
        output_dir = os.path.join(output_dir, 'language_model')
        os.makedirs(output_dir, exist_ok=True)
        logger.info(f"Saving language_model checkpoint to {output_dir}")

        # supported_classes = (PreTrainedModel,) if not is_peft_available() else (PreTrainedModel, PeftModel)
        # Save a trained model and configuration using `save_pretrained()`.
        # They can then be reloaded using `from_pretrained()`
        self.model.language_model.save_pretrained(
                output_dir, state_dict=None, safe_serialization=self.args.save_safetensors
            )


def get_scheduler(
    name: Union[str, SchedulerType],
    optimizer: Optimizer,
    num_warmup_steps: Optional[int] = None,
    num_training_steps: Optional[int] = None,
    patience: Optional[int] = 0,
    mode: str = 'max',
    factor: float = 0.3,
):
    name = SchedulerType(name)
    schedule_func = TYPE_TO_SCHEDULER_FUNCTION[name]
    if name == SchedulerType.CONSTANT:
        return schedule_func(optimizer)
    
    if name == SchedulerType.REDUCE_ON_PLATEAU:
        return ReduceLROnPlateau(optimizer, patience=patience, mode=mode, factor=factor)

    # All other schedulers require `num_warmup_steps`
    if num_warmup_steps is None:
        raise ValueError(f"{name} requires `num_warmup_steps`, please provide that argument.")

    if name == SchedulerType.CONSTANT_WITH_WARMUP:
        return schedule_func(optimizer, num_warmup_steps=num_warmup_steps)

    if name == SchedulerType.INVERSE_SQRT:
        return schedule_func(optimizer, num_warmup_steps=num_warmup_steps)

    # All other schedulers require `num_training_steps`
    if num_training_steps is None:
        raise ValueError(f"{name} requires `num_training_steps`, please provide that argument.")

    return schedule_func(optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_training_steps)


class TrainerWithCustomizedPatience(Trainer):

    def create_scheduler(self, num_training_steps: int, optimizer: torch.optim.Optimizer = None):
        """
        Setup the scheduler. The optimizer of the trainer must have been set up either before this method is called or
        passed as an argument.

        Args:
            num_training_steps (int): The number of training steps to do.
        """
        if self.lr_scheduler is None:
            self.lr_scheduler = get_scheduler(
                self.args.lr_scheduler_type,
                optimizer=self.optimizer if optimizer is None else optimizer,
                num_warmup_steps=self.args.get_warmup_steps(num_training_steps),
                num_training_steps=num_training_steps,
                patience=self.args.patience,
                mode='max' if self.args.greater_is_better else 'min',
                factor=self.args.factor,
            )
            self._created_lr_scheduler = True
        return self.lr_scheduler
    
    # def create_optimizer(self):
    #     """
    #     Setup the optimizer.

    #     We provide a reasonable default that works well. If you want to use something else, you can pass a tuple in the
    #     Trainer's init through `optimizers`, or subclass and override this method in a subclass.
    #     """
    #     opt_model = self.model_wrapped if is_sagemaker_mp_enabled() else self.model

    #     if self.optimizer is None:
    #         decay_parameters = self.get_decay_parameter_names(opt_model)
    #         optimizer_grouped_parameters = [
    #             {
    #                 "params": [
    #                     p for n, p in opt_model.named_parameters() if (n in decay_parameters and p.requires_grad) and \
    #                                                                     'language_model' not in n
    #                 ],
    #                 "lr": self.args.learning_rate,
    #                 "weight_decay": self.args.weight_decay,
    #             },
    #             {
    #                 "params": [
    #                     p for n, p in opt_model.named_parameters() if (n not in decay_parameters and p.requires_grad) and \
    #                                                                     'language_model' not in n
    #                 ],
    #                 "lr": self.args.learning_rate,
    #                 "weight_decay": 0.0,
    #             },
    #             {
    #                 "params": [
    #                     p for n, p in opt_model.named_parameters() if (n in decay_parameters and p.requires_grad) and \
    #                                                                     ('language_model' in n)
    #                 ],
    #                 "lr": self.model.config_dict['language_model']['learning_rate'],
    #                 "weight_decay": self.model.config_dict['language_model']['weight_decay'],
    #             },
    #             {
    #                 "params": [
    #                     p for n, p in opt_model.named_parameters() if (n not in decay_parameters and p.requires_grad) and \
    #                                                                     ('language_model' in n)
    #                 ],
    #                 "lr": self.model.config_dict['language_model']['learning_rate'],
    #                 "weight_decay": 0.0,
    #             },
    #         ]

    #         optimizer_cls, optimizer_kwargs = Trainer.get_optimizer_cls_and_kwargs(self.args)

    #         self.optimizer = optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs)
    #         if optimizer_cls.__name__ == "Adam8bit":
    #             import bitsandbytes

    #             manager = bitsandbytes.optim.GlobalOptimManager.get_instance()

    #             skipped = 0
    #             for module in opt_model.modules():
    #                 if isinstance(module, nn.Embedding):
    #                     skipped += sum({p.data_ptr(): p.numel() for p in module.parameters()}.values())
    #                     logger.info(f"skipped {module}: {skipped/2**20}M params")
    #                     manager.register_module_override(module, "weight", {"optim_bits": 32})
    #                     logger.debug(f"bitsandbytes: will optimize {module} in fp32")
    #             logger.info(f"skipped: {skipped/2**20}M params")

    #     if is_sagemaker_mp_enabled():
    #         self.optimizer = smp.DistributedOptimizer(self.optimizer)

    #     return self.optimizer
    





class ContrastiveTrainerMix(ContrastiveTrainer):

    def _get_train_sampler(self):

        return DomainSampler(
                    dataset=self.train_dataset, 
                    batch_size=self.args.train_batch_size, 
                    iter_dict=self.args.iter_dict,
                    scale_dict=self.args.scale_dict,
                    shuffle=True)
    
    def _get_eval_sampler(self, eval_dataset: Dataset):
        return DomainSampler(
                    dataset=eval_dataset, 
                    batch_size=self.args.eval_batch_size, 
                    iter_dict=self.args.iter_dict,
                    scale_dict=self.args.scale_dict,
                    shuffle=False)
        