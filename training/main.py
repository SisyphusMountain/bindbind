from lightning.pytorch.loggers import WandbLogger
import lightning.pytorch as pl
import torch
from torch_geometric.loader import DataLoader
import sys
sys.path.append("/fs/pool/pool-marsot/")
import logging
import hydra
from bindbind.models.model import TankBindModel
from tankbind_enzo.bind.ML.modules.model import TankBindModel as old_TankBindModel 
from bindbind.torch_datasets.tankbind_dataset import TankBindDataset, TankBindDatasetWithoutp2rank
from bindbind.tests.debug import TankBindDebugDataset
from bindbind.training.sampler import TankBindSampler 
from bindbind.torch_datasets.tankbind_dataloader import TankBindDataLoader
from lightning.pytorch.callbacks import Callback
from bindbind.training.callbacks import get_callbacks
import logging 
from omegaconf import OmegaConf

logger = logging.getLogger(__name__)
logger.setLevel(logging.WARN)

@hydra.main(version_base=None,
            config_path="/fs/pool/pool-marsot/bindbind/training/configs",)
def main(cfg):
    torch.set_float32_matmul_precision("high")
    pl.seed_everything(cfg.seed)
    logger.info(OmegaConf.to_yaml(cfg))
    wandb_logger = WandbLogger(project='protein_binding', log_model=False,
                               save_dir=cfg.logs.save_dir,)
    csv_logger = pl.loggers.CSVLogger(save_dir=cfg.logs.save_dir,
                                      )
    if cfg.tankbind.old_model:
        model = old_TankBindModel()
    else:
        model = TankBindModel(cfg=cfg,
                        esm_features = cfg.ablations.esm_features,
                        use_gvp=cfg.ablations.use_gvp,
                        logger=wandb_logger,
                        fast_attention=cfg.ablations.fast_attention,)

    if not cfg.debug:
        dataset = TankBindDatasetWithoutp2rank(add_esm_embeddings=cfg.ablations.esm_features, noise_range=cfg.training.add_noise, contact_threshold=cfg.tankbind.contact_threshold, pocket_radius=cfg.tankbind.pocket_radius)
    elif cfg.debug:
        dataset = TankBindDebugDataset()


    if not cfg.debug:
        with open(cfg.splits.train_path, "r") as f:
            train_indices = set([name for name in f.read().split("\n")])
        with open(cfg.splits.val_path, "r") as f:
            valid_indices = set([name for name in f.read().split("\n")])
        train_dataset_indices = (dataset.proteins_df[dataset.proteins_df["protein_names"].isin(train_indices)].index).tolist()
        val_dataset_indices = (dataset.proteins_df[dataset.proteins_df["protein_names"].isin(valid_indices)].index).tolist()
        train = dataset[train_dataset_indices]
        val = dataset[val_dataset_indices]
        val.noise_range=0.0
    elif cfg.debug:
        train = dataset
    
    if cfg.training.adaptative_batch_size:
        train_sampler = TankBindSampler(train,
                                        max_num = cfg.training.max_batch_size,
                                        available_memory_gb=cfg.training.available_memory_gb,
                                        shuffle = True,
                                        )
        val_sampler = TankBindSampler(val,
                                        max_num = cfg.training.max_batch_size,
                                        available_memory_gb=cfg.training.available_memory_gb,
                                        shuffle = False,
        )
        train_dataloader = TankBindDataLoader(train,
                                              batch_sampler=train_sampler,
                                              follow_batch=["ligand_in_pocket_mask"],
                                              shuffle=False,
                                              num_workers=cfg.num_workers,
                                              make_divisible_by_8=cfg.ablations.make_input_dims_divisible_by_8)
        val_dataloader = TankBindDataLoader(val,
                                            batch_sampler=val_sampler,
                                            shuffle=False,
                                            follow_batch=["ligand_in_pocket_mask"],
                                            num_workers=cfg.num_workers,
                                            make_divisible_by_8=cfg.ablations.make_input_dims_divisible_by_8)
    else:
        torch.manual_seed(1)
        G = torch.Generator()
        G.manual_seed(1)
        train_dataloader = TankBindDataLoader(train, batch_size=cfg.training.batch_size, follow_batch=["ligand_in_pocket_mask"], shuffle=True, num_workers=cfg.num_workers, pin_memory=True, pin_memory_device="cuda:0", generator=G)
        val_dataloader = TankBindDataLoader(val, batch_size=cfg.training.batch_size, follow_batch=["ligand_in_pocket_mask"], shuffle=False, num_workers=cfg.num_workers, pin_memory=True, pin_memory_device="cuda:0", generator=G)
    
    if cfg.ablations.pretraining:
        pretrain_dataset = TankBindDataset(add_esm_embeddings=cfg.ablations.esm_features, noise_range=cfg.training.add_noise, contact_threshold=cfg.tankbind.contact_threshold)
        pretrain_dataset_indices = (pretrain_dataset.pockets_df['name'].isin(train_indices).index).tolist()
        pretrain_dataset = pretrain_dataset[pretrain_dataset_indices]
        pretrain_dataloader = TankBindDataLoader(pretrain_dataset,
                                                 batch_size=cfg.training.batch_size,
                                                 follow_batch=["ligand_in_pocket_mask"],
                                                 shuffle=True,
                                                 num_workers=cfg.num_workers,
                                                 make_divisible_by_8=cfg.ablations.make_input_dims_divisible_by_8)
        pre_trainer = pl.Trainer(max_epochs=cfg.training.pretraining_epochs,
                                    logger=wandb_logger,
                                    log_every_n_steps=100,
                                    accelerator="auto",
                                    precision=cfg.training.precision,
                                    callbacks=get_callbacks(cfg,
                                                            train,
                                                            swa=cfg.training.swa.activate,
                                                            log_memory_allocated=cfg.training.log_memory_allocated,
                                                            evaluation=cfg.training.evaluation,
                                                            empty_cache=cfg.training.empty_cache,
                                                            log_every_n_epochs=cfg.training.log_every_n_epochs,
                                                            ),
                                    )
        pre_trainer.fit(model, pretrain_dataloader, val_dataloader,)
        del pretrain_dataloader
        del pretrain_dataset_indices
        del pretrain_dataset

    trainer = pl.Trainer(max_epochs=cfg.training.epochs,
                               logger=[wandb_logger, csv_logger],
                               log_every_n_steps=100,
                               accelerator="auto",
                               precision=cfg.training.precision,
                               callbacks=get_callbacks(cfg,
                                                       train,
                                                       swa=cfg.training.swa.activate,
                                                       log_memory_allocated=cfg.training.log_memory_allocated,
                                                       evaluation=cfg.training.evaluation,
                                                       empty_cache=cfg.training.empty_cache,
                                                       log_every_n_epochs=cfg.training.log_every_n_epochs,
                                                       ),
                               gradient_clip_val=cfg.training.gradient_clip_val,
                               )
    
    
    # See documentation: to profile training, we can use pytorch_lightning profiles: trainer = Trainer(profiler="advanced")
    trainer.fit(model, train_dataloader, val_dataloader,)


if __name__ == "__main__":
    main()
