import dataclasses
import os
from typing import Literal

import matplotlib.pyplot as plt
import mrcfile
import numpy as np
import pytorch_lightning as pl
import rich
import torch
import torch.nn as nn
import torch.nn.functional as F
import tyro
from einops import rearrange, repeat
from pytorch_lightning.callbacks import (ModelCheckpoint, ProgressBar,
                                         RichProgressBar, TQDMProgressBar)
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.utilities.model_summary import ModelSummary
from rich.progress import track
from torch.utils.data import DataLoader

from code.dataset import EMPIARDataset
from code.model import CryoNeRF


@dataclasses.dataclass
class Args:
    """Arguments of CryoNeRF."""
    
    size: int = 256
    """Size of the volume and particle images."""

    batch_size: int = 1
    """Batch size for training."""
    
    ray_num: int = 1024
    """Number of rays to query in a batch."""
    
    nerf_enc_dim: int = 16
    """Positional encoding dim for sin and cos, the output dim is 6 * enc_dim."""
    
    nerf_hid_dim: int = 160
    """Hidden dim of NeRF."""
    
    nerf_hid_layer_num: int = 2
    """Number of hidden layers besides the input and output layer."""
    
    dfom_enc_dim: int = 16
    """Positional encoding dim for sin and cos, the output dim is 6 * enc_dim."""
    
    dfom_hid_dim: int = 160
    """Hidden dim of deformation field."""
    
    dfom_hid_layer_num: int = 2
    """Number of hidden layers besides the input and output layer of deformation field.."""
    
    dfom_encoder: Literal["resnet18", "resnet34"] = "resnet18"
    """Encoder for deformation latent variable."""
    
    dfom_latent_dim: int = 16
    """Latent variable dim for deformation encoder."""
    
    save_dir: str = "experiments/test"
    """Dir to save visualization and checkpoint."""
    
    log_step: int = 1000
    """Number of steps to log once."""
    
    print_step: int = 100
    """Number of steps to print once."""
    
    dataset: Literal["empiar-10028", "empiar-10076"] = "empiar-10028"
    """EMPIAR dataset to use."""
    
    root_dir: str = "/data/workspace/huaizhi"
    """Root dir for datasets."""
    
    sign: Literal[1, -1] = 1
    """Sign of the particle images."""
    
    seed: int = -1
    """Seed everything"""
    
    load_ckpt: str = ""
    """The checkpoint to load"""
    
    epochs: int = 1
    """Number of epochs for training."""
    
    enable_dfom: bool = False
    """Whether to enable deformation for heterogeneous reconstruction."""
    
    checkpointing: bool = False
    """Whether to use checkpointing to save GPU memory."""
    
    val_only: bool = False
    """Only val"""
    
    test_only: bool = False
    """Only test"""
    
    
class IterationProgressBar(TQDMProgressBar):
    def init_train_tqdm(self):
        bar = super().init_train_tqdm()
        if self.trainer.max_steps:
            bar.total = self.trainer.max_steps
        else:
            bar.total = self.trainer.num_training_batches
        return bar

    def on_train_epoch_start(self, trainer, pl_module):
        # Only reset if max_steps is not set
        if not self.trainer.max_steps:
            super().on_train_epoch_start(trainer, pl_module)

    def init_validation_tqdm(self):
        bar = super().init_validation_tqdm()
        bar.total = self.trainer.num_val_batches[0] 
        return bar
    
    
class RichIterationProgressBar(RichProgressBar):
        def on_train_epoch_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
            if self.is_disabled:
                return
            total_batches = self.total_train_batches
            train_description = "Training..."

            if self.train_progress_bar_id is not None and self._leave:
                self._stop_progress()
                self._init_progress(trainer)
            if self.progress is not None:
                if self.train_progress_bar_id is None:
                    self.train_progress_bar_id = self._add_task(total_batches, train_description)
                else:
                    self.progress.reset(
                        self.train_progress_bar_id,
                        total=total_batches,
                        description=train_description,
                        visible=True,
                    )

            self.refresh()
            
        def get_metrics(self, trainer, model):
            # don't show the version number
            items = super().get_metrics(trainer, model)
            items.pop("v_num", None)
            return items
    

if __name__ == "__main__":
    args = tyro.cli(Args)
    
    if args.seed != -1:
        pl.seed_everything(args.seed)
    
    os.makedirs(args.save_dir, exist_ok=True)
    
    if args.dataset == "empiar-10028":
        sign = -1
    elif args.dataset == "empiar-10076":
        sign = 1
    else:
        sign = None
        rich.print("[red]Unknown dataset. Use sign specified in args![/red]")
    
    if args.load_ckpt:
        cryo_nerf = CryoNeRF.load_from_checkpoint(checkpoint_path=args.load_ckpt, strict=False, args=args)
        print("Model loaded:", args.load_ckpt)
    else:
        cryo_nerf = CryoNeRF(args)
        
    dataset = EMPIARDataset(
        mrcs=f"{args.root_dir}/{args.dataset}/particles.mrcs",
        ctf=f"{args.root_dir}/{args.dataset}/ctf.pkl",
        poses=f"{args.root_dir}/{args.dataset}/poses.pkl",
        size=args.size, sign=sign if sign is not None else args.sign,
    )

    train_dataloader = DataLoader(dataset, batch_size=args.batch_size, num_workers=8, shuffle=False, pin_memory=True, drop_last=True)
    valid_dataloader = DataLoader(dataset, batch_size=16, num_workers=8, shuffle=False, pin_memory=True, drop_last=True)
        
    logger = WandbLogger(name=f"CryoNeRF-{args.save_dir}", save_dir=args.save_dir, offline=True)
    logger.experiment.log_code(".")
    
    checkpoint_callback = ModelCheckpoint(dirpath=args.save_dir, save_top_k=-1, verbose=True, every_n_train_steps=10000, save_last=True)
    
    if not args.load_ckpt:
        assert not args.val_only and not args.test_only, "Must specify a ckpt for val or test!!!"
        init_trainer = pl.Trainer(accelerator="gpu", devices=1, max_steps=1, precision="16-mixed", barebones=True)
        init_trainer.fit(model=cryo_nerf, train_dataloaders=train_dataloader)
        init_trainer.save_checkpoint(f"{args.save_dir}/temp.ckpt")
        del cryo_nerf
        del init_trainer
    
    trainer = pl.Trainer(
        accelerator="gpu",
        strategy="ddp_find_unused_parameters_true",  # "ddp", "deepspeed", "ddp_find_unused_parameters_true"
        max_epochs=args.epochs,
        # max_steps=1000,
        logger=logger,
        callbacks=[RichIterationProgressBar(), checkpoint_callback],
        precision="16-mixed",
    )
    validator = pl.Trainer(
        accelerator="gpu",
        strategy="ddp_find_unused_parameters_true",  # "ddp", "deepspeed", "ddp_find_unused_parameters_true"
        max_epochs=args.epochs,
        logger=None,
        enable_checkpointing=False,
        enable_model_summary=False,
        devices=1,
        callbacks=[RichIterationProgressBar()],
        precision="16-mixed",
    )
    
    if args.val_only:
        validator.validate(model=cryo_nerf, dataloaders=valid_dataloader)
    elif args.test_only:
        pass
    else:
        if not args.load_ckpt:
            cryo_nerf = CryoNeRF.load_from_checkpoint(f"{args.save_dir}/temp.ckpt", strict=False, args=args)
        trainer.fit(model=cryo_nerf, train_dataloaders=train_dataloader)
        validator.validate(model=cryo_nerf, dataloaders=valid_dataloader)
