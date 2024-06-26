from bindbind.training.sampler import DynamicSamplerProgressBar
from bindbind.torch_datasets.tankbind_dataset import NoisyCoordinates
from bindbind.experiments.ablations.regular.metrics.metrics_fast import evaluate_model_val
from lightning.pytorch.callbacks import DeviceStatsMonitor, ModelCheckpoint, StochasticWeightAveraging

import torch
from lightning.pytorch.callbacks import Callback
import GPUtil
from typing import Optional, Tuple, Union, Any
from lightning.pytorch import Trainer, LightningModule
from lightning.pytorch.utilities.types import STEP_OUTPUT


def get_callbacks(cfg,
                  train,
                  log_memory_allocated=False,
                  swa=True,
                  evaluation=True,
                  empty_cache=False,
                  log_every_n_epochs=10):
    callbacks = []
    if cfg.training.adaptative_batch_size:
        callbacks.append(DynamicSamplerProgressBar(len(train)))
    callbacks.append(NoisyCoordinates(train))
    if log_memory_allocated:
        callbacks.append(GPUMaxUtilizationCallback())
    callbacks.append(DeviceStatsMonitor())
    callbacks.append(ModelCheckpoint(every_n_epochs=1, save_top_k=-1))
    if swa:
        callbacks.append(StochasticWeightAveraging(swa_lrs=cfg.swa.lr))
    if empty_cache:
        callbacks.append(EmptyCache())
    if evaluation:
        callbacks.append(EvaluationCallback(log_every_n_epochs=log_every_n_epochs))
    return callbacks

class GPUMaxUtilizationCallback(Callback):
    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        torch.cuda.reset_max_memory_allocated()
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        self.log_dict({"max memory":torch.cuda.max_memory_allocated()/1024**3})

def compute_gpu_usage(
    timestamp_label: Optional[str] = None, return_max: bool = False
) -> Optional[Union[float, Tuple[float, float]]]:
    gpus = GPUtil.getGPUs()
    total, free = torch.cuda.mem_get_info()
    if gpus and torch.cuda.device_count() == 1:
        vram_usage = gpus[0].memoryUtil * 100  # assuming single GPU
        vram_max = (torch.cuda.max_memory_allocated() / total) * 100
    elif gpus and torch.cuda.device_count() > 1:  # still needs testing
        vram_usage = sum([gpu.memoryUtil for gpu in gpus]) / torch.cuda.device_count()
        vram_max = (torch.cuda.max_memory_allocated() / total) * 100
    else:
        vram_usage = 0
    if timestamp_label:
        print(
            f"Timestamp: {timestamp_label} -- VRAM usage: {vram_usage} -- Max {vram_max}"
        )
    else:
        if return_max:
            return vram_usage, vram_max
        else:
            return vram_usage

class EmptyCache(Callback):
    def __init__(self, verbose: bool = False) -> None:
        super().__init__()
        self.verbose = verbose
    def on_train_batch_start(
        self, trainer: Trainer, pl_module: LightningModule, batch: Any, batch_idx: int
    ) -> None:
        if self.verbose:
            step = "on_train_batch_start"
            vram_before = compute_gpu_usage()
            torch.cuda.empty_cache()
            vram_after = compute_gpu_usage()
            pl_module.log(f"{step}_before", vram_before)
            pl_module.log(f"{step}_after", vram_after)
        else:
            torch.cuda.empty_cache()
        return None
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        torch.cuda.empty_cache()
        return None
    def on_train_epoch_end(self, trainer, pl_module) -> None:
        if self.verbose:
            step = "on_train_epoch_end"
            vram_before = compute_gpu_usage()
            torch.cuda.empty_cache()
            vram_after = compute_gpu_usage()
            pl_module.log(f"{step}_before", vram_before)
            pl_module.log(f"{step}_after", vram_after)
        else:
            torch.cuda.empty_cache()
        return None
    def on_validation_epoch_start(self, trainer, pl_module) -> None:
        if self.verbose:
            step = "on_validation_epoch_start"
            vram_before = compute_gpu_usage()
            torch.cuda.empty_cache()
            vram_after = compute_gpu_usage()
            pl_module.log(f"{step}_before", vram_before)
            pl_module.log(f"{step}_after", vram_after)
        else:
            torch.cuda.empty_cache()
        return None
    def on_validation_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: STEP_OUTPUT,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        if self.verbose:
            step = "on_validation_batch_end"
            vram_before = compute_gpu_usage()
            torch.cuda.empty_cache()
            vram_after = compute_gpu_usage()
            pl_module.log(f"{step}_before", vram_before)
            pl_module.log(f"{step}_after", vram_after)
        else:
            torch.cuda.empty_cache()
        return None
    def on_validation_epoch_end(self, trainer, pl_module) -> None:
        if self.verbose:
            step = "on_validation_epoch_end"
            vram_before = compute_gpu_usage()
            torch.cuda.empty_cache()
            vram_after = compute_gpu_usage()
            pl_module.log(f"{step}_before", vram_before)
            pl_module.log(f"{step}_after", vram_after)
        else:
            torch.cuda.empty_cache()
        return None
    def on_validation_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        if self.verbose:
            step = "on_validation_end"
            vram_before = compute_gpu_usage()
            torch.cuda.empty_cache()
            vram_after = compute_gpu_usage()
            print(f"{step}_before: {vram_before}")
            print(f"{step}_after: {vram_after}")
        else:
            torch.cuda.empty_cache()
        return None
    

class EvaluationCallback(Callback):
    def __init__(self, log_every_n_epochs: int = 5) -> None:
        self.log_every_n_epochs = log_every_n_epochs
    def on_train_epoch_end(self, trainer, pl_module):
        if trainer.current_epoch % self.log_every_n_epochs == self.log_every_n_epochs - 1:
            df = evaluate_model_val(pl_module, batch_size=8)
            dict_rmsd, dict_com_dist = df[["mean", '25%', '50%', '75%', '5A', '2A', 'median']].to_dict(orient="records")
            pl_module.log_dict(
                dict_rmsd
            )
            pl_module.log_dict(
                dict_com_dist
            )
            print(df)
