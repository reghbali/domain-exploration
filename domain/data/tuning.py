import os
import sys

root_dir = 'system/root/directory'
sys.path.append(root_dir)

import argparse
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning.pytorch as pl
import torch.optim as optim
import optuna
from optuna.integration import PyTorchLightningPruningCallback
from torchvision import datasets, transforms
from domain.model.modules import MLP, ResidualMLP, MLP_Block_Residual, ResCNN, CNN, ResCNN_block, VisionTransformer
from domain.data.data_modules import MNISTDataModule, CIFAR10DataModule, ImageNetDataModule
from domain.model.lit_modules import LitClassificationModel as LiT

PERCENT_VALID_EXAMPLES = 0.1
BATCHSIZE = 128
EPOCHS = 10
DIR = root_dir


def objective(trial: optuna.trial.Trial, model: nn.Module, datamodule: pl.LightningDataModule) -> float:
    model = LiT(model)

    trainer = pl.Trainer(
        logger=True,
        limit_val_batches=PERCENT_VALID_EXAMPLES,
        enable_checkpointing=False,
        max_epochs=EPOCHS,
        accelerator="auto",
        devices=1,
        callbacks=[PyTorchLightningPruningCallback(trial, monitor="val_acc")],
    )
    
    trainer.fit(model, datamodule=datamodule)

    return trainer.callback_metrics["val_acc"].item()

def get_args():
    parser = argparse.ArgumentParser(description="Training script")
    parser.add_argument('--pruning', action='store_true', help='Enable pruning')
    parser.add_argument('--architecture', type=str, required=True, help='Model architecture (mlp, skipmlp, cnn, rescnn, vit)')
    parser.add_argument('--domain', type=str, required=True, help='Domain (frequency or pixel)')
    parser.add_argument('--dataset', type=str, required=True, help='Dataset (MNIST, CIFAR10, ImageNet)')
    
    if 'ipykernel_launcher' in sys.argv[0]:
        args, unknown = parser.parse_known_args()
    else:
        args = parser.parse_args()
    return args

def build_model(trial: optuna.trial.Trial, architecture: str, domain: str, dataset: str) -> nn.Module:
    if architecture == "mlp":
        hidden_factor = trial.suggest_int("hidden_factor", 1, 10)
        depth = trial.suggest_int("depth", 1, 5)
        model = MLP(input_shape=(28, 28), num_classes=10, hidden_factor=hidden_factor, depth=depth)
    
    elif architecture == "skipmlp":
        layers = tuple(trial.suggest_int(f"layer_{i}", 32, 128) for i in range(trial.suggest_int("depth", 1, 5)))
        model = ResidualMLP(input_shape=(28, 28, 1), layers=layers, num_classes=10)

    elif architecture == "cnn":
        conv1_feature_maps = trial.suggest_int("conv1_feature_maps", 16, 64)
        conv2_feature_maps = trial.suggest_int("conv2_feature_maps", 32, 128)
        kernel_size = trial.suggest_int("kernel_size", 3, 7)
        pool_size = trial.suggest_int("pool_size", 2, 3)
        linear_size1 = trial.suggest_int("linear_size1", 128, 512)
        linear_size2 = trial.suggest_int("linear_size2", 64, 256)
        linear_size3 = trial.suggest_int("linear_size3", 32, 128)
        model = CNN(input_shape=(28, 28), num_classes=10, in_channels=1, conv1_feature_maps=conv1_feature_maps,
                    conv2_feature_maps=conv2_feature_maps, kernel_size=kernel_size, pool_size=pool_size,
                    linear_size1=linear_size1, linear_size2=linear_size2, linear_size3=linear_size3, activation=nn.GELU)
    
    elif architecture == "rescnn":
        layers = (trial.suggest_int("layer1", 2, 5), trial.suggest_int("layer2", 2, 5), 
                  trial.suggest_int("layer3", 2, 5), trial.suggest_int("layer4", 2, 5))
        model = ResCNN(ResCNN_block, layers=layers, in_channels=1, num_classes=10)
    
    elif architecture == "vit":
        if dataset == 'CIFAR10':
            patch_size = trial.suggest_categorical("patch_size", [8])
            embed_dim = trial.suggest_categorical("embed_dim", [256])
            num_attention_heads = trial.suggest_categorical("num_attention_heads", [8])
            num_layers = trial.suggest_categorical("num_layers", [8])
            attention_head_dim = trial.suggest_categorical("attention_head_dim", [32])
            mlp_head_dim = trial.suggest_categorical("mlp_head_dim", [1024])
            input_shape = (3, 32, 32)
            in_channels = 3
            num_classes = 10
            
        elif dataset == 'MNIST':
            patch_size = trial.suggest_categorical("patch_size", [7])
            embed_dim = trial.suggest_categorical("embed_dim", [256])
            num_attention_heads = trial.suggest_categorical("num_attention_heads", [8])
            num_layers = trial.suggest_categorical("num_layers", [8])
            attention_head_dim = trial.suggest_categorical("attention_head_dim", [32])
            mlp_head_dim = trial.suggest_categorical("mlp_head_dim", [1024])
            input_shape = (1, 28, 28)
            in_channels = 1
            num_classes = 10
            
        elif dataset == 'ImageNet':
            patch_size = trial.suggest_categorical("patch_size", [16])
            embed_dim = trial.suggest_categorical("embed_dim", [768])
            num_attention_heads = trial.suggest_categorical("num_attention_heads", [12])
            num_layers = trial.suggest_categorical("num_layers", [12])
            attention_head_dim = trial.suggest_categorical("attention_head_dim", [64])
            mlp_head_dim = trial.suggest_categorical("mlp_head_dim", [3072])
            input_shape = (3, 224, 224)
            in_channels = 3
            num_classes = 1000

        model = VisionTransformer(
            input_shape=input_shape,
            in_channels=in_channels,
            num_classes=num_classes,
            dropout_rate=0.1,
            patch_size=patch_size,
            embed_dim=embed_dim,
            num_attention_heads=num_attention_heads,
            num_layers=num_layers,
            attention_head_dim=attention_head_dim,
            mlp_head_dim=mlp_head_dim,
            freq=False
        )
                                  
    return model

def build_datamodule(dataset: str, domain: str) -> pl.LightningDataModule:
    if dataset == 'MNIST':
        datamodule = MNISTDataModule(domain=domain, batch_size=BATCHSIZE)
    elif dataset == 'CIFAR10':
        datamodule = CIFAR10DataModule(domain=domain, batch_size=BATCHSIZE)
    elif dataset == 'ImageNet':
        datamodule = ImageNetDataModule(domain=domain, batch_size=BATCHSIZE)
    return datamodule

if __name__ == "__main__":
    args = get_args()

    pruner = optuna.pruners.MedianPruner() if args.pruning else optuna.pruners.NopPruner()

    study = optuna.create_study(direction="maximize", pruner=pruner)
    study.optimize(lambda trial: objective(trial, build_model(trial, args.architecture, args.domain,args.dataset), build_datamodule(args.dataset, args.domain)), n_trials=2, timeout=3600)

    print("Number of finished trials: {}".format(len(study.trials)))

    print("Best trial:")
    trial = study.best_trial

    print("  Value: {}".format(trial.value))

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))