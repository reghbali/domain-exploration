import argparse
from collections import namedtuple
import sys

import torch.nn as nn
import lightning.pytorch as pl
import optuna
from optuna.integration import PyTorchLightningPruningCallback
from domain.model.modules import (
    MLP,
    ResidualMLP,
    ResCNN,
    CNN,
    ResCNN_block,
    VisionTransformer,
)
from domain.data.data_modules import (
    MNISTDataModule,
    CIFAR10DataModule,
    ImageNetDataModule,
)
from domain.model.lit_modules import LitClassificationModel as LiT

DatasetInfo = namedtuple('DatasetInfo', ['name', 'image_shape', 'num_classes'])
dataset_info_table = {
    'MNIST': DatasetInfo('MNIST', (28, 28, 1), 10),
    'CIFAR10': DatasetInfo('CIFAR10', (32, 32, 3), 10),
    'ImageNet': DatasetInfo('ImageNet', (224, 224, 3), 1000),
}

valid_examples = 0.1


def objective(
    trial: optuna.trial.Trial,
    model: nn.Module,
    datamodule: pl.LightningDataModule,
    epochs: int,
    name,
) -> float:
    model = LiT(model, datamodule.num_classes)

    trainer = pl.Trainer(
        logger=True,
        default_root_dir=name,
        limit_val_batches=valid_examples,
        enable_checkpointing=False,
        max_epochs=epochs,
        accelerator='auto',
        devices=1,
        callbacks=[PyTorchLightningPruningCallback(trial, monitor='val_acc')],
    )

    trainer.fit(model, datamodule=datamodule)

    return trainer.callback_metrics['val_acc'].item()


def get_args():
    parser = argparse.ArgumentParser(description='Training script')
    parser.add_argument('--pruning', action='store_true', help='Enable pruning')
    parser.add_argument(
        '--architecture',
        type=str,
        required=True,
        help='Model architecture (mlp, skipmlp, cnn, rescnn, vit)',
    )
    parser.add_argument(
        '--domain', type=str, required=True, help='Domain (frequency or pixel)'
    )
    parser.add_argument(
        '--dataset', type=str, required=True, help='Dataset (MNIST, CIFAR10, ImageNet)'
    )
    parser.add_argument('--batchsize', type=int, required=True, help='Batch size')
    parser.add_argument('--epochs', type=int, required=True, help='Number of epochs')

    if 'ipykernel_launcher' in sys.argv[0]:
        args, unknown = parser.parse_known_args()
    else:
        args = parser.parse_args()
    return args


def build_model(
    trial: optuna.trial.Trial, architecture: str, domain: str, dataset_info: DatasetInfo
) -> nn.Module:
    if architecture == 'mlp':
        hidden_factor = trial.suggest_int('hidden_factor', 1, 3)
        depth = trial.suggest_int('depth', 1, 3)
        model = MLP(
            input_shape=dataset_info.image_shape,
            num_classes=dataset_info.num_classes,
            hidden_factor=hidden_factor,
            depth=depth,
        )

    elif architecture == 'skipmlp':
        layers = tuple(
            2 ** trial.suggest_int(f'layer_{i}', 6, 8)
            for i in range(trial.suggest_int('depth', 1, 4))
        )
        model = ResidualMLP(
            input_shape=dataset_info.image_shape, layers=layers, num_classes=10
        )

    elif architecture == 'cnn':
        root = trial.suggest_int('root', 16, 64)
        conv1_feature_maps = root
        conv2_feature_maps = root * 2
        kernel_size = max(3, root // 16)
        pool_size = 2 if root < 48 else 3
        linear_size1 = root * 8
        linear_size2 = root * 4
        linear_size3 = root * 2
        model = CNN(
            input_shape=dataset_info.image_shape[:-1],
            num_classes=dataset_info.num_classes,
            in_channels=dataset_info.image_shape[-1],
            conv1_feature_maps=conv1_feature_maps,
            conv2_feature_maps=conv2_feature_maps,
            kernel_size=kernel_size,
            pool_size=pool_size,
            linear_size1=linear_size1,
            linear_size2=linear_size2,
            linear_size3=linear_size3,
            activation=nn.GELU,
        )

    elif architecture == 'rescnn':
        layers = (
            trial.suggest_int('layer1', 2,5),
            trial.suggest_int('layer2', 2,5),
            trial.suggest_int('layer3', 3,6),
            trial.suggest_int('layer4', 2,4),
        )
        model = ResCNN(ResCNN_block, layers=layers, in_channels=1, num_classes=10)

    elif architecture == 'vit':
        if dataset_info.name == 'CIFAR10':
            patch_size = trial.suggest_categorical('patch_size', [8])
            embed_dim = trial.suggest_categorical('embed_dim', [256])
            num_attention_heads = trial.suggest_categorical('num_attention_heads', [8])
            num_layers = trial.suggest_categorical('num_layers', [8])
            attention_head_dim = trial.suggest_categorical('attention_head_dim', [32])
            mlp_head_dim = trial.suggest_categorical('mlp_head_dim', [1024])

        elif dataset_info.name == 'MNIST':
            patch_size = trial.suggest_categorical('patch_size', [7])
            embed_dim = trial.suggest_categorical('embed_dim', [256])
            num_attention_heads = trial.suggest_categorical('num_attention_heads', [8])
            num_layers = trial.suggest_categorical('num_layers', [8])
            attention_head_dim = trial.suggest_categorical('attention_head_dim', [32])
            mlp_head_dim = trial.suggest_categorical('mlp_head_dim', [1024])

        elif dataset_info.name == 'ImageNet':
            patch_size = trial.suggest_categorical('patch_size', [16])
            embed_dim = trial.suggest_categorical('embed_dim', [768])
            num_attention_heads = trial.suggest_categorical('num_attention_heads', [12])
            num_layers = trial.suggest_categorical('num_layers', [12])
            attention_head_dim = trial.suggest_categorical('attention_head_dim', [64])
            mlp_head_dim = trial.suggest_categorical('mlp_head_dim', [3072])

        model = VisionTransformer(
            input_shape=dataset_info.image_shape,
            in_channels=dataset_info.image_shape[-1],
            num_classes=dataset_info.num_classes,
            dropout_rate=0.1,
            patch_size=patch_size,
            embed_dim=embed_dim,
            num_attention_heads=num_attention_heads,
            num_layers=num_layers,
            attention_head_dim=attention_head_dim,
            mlp_head_dim=mlp_head_dim,
            freq=False,
        )

    return model


def build_datamodule(
    dataset: str, domain: str, batch_size: int
) -> pl.LightningDataModule:
    if dataset == 'MNIST':
        datamodule = MNISTDataModule(domain=domain, batch_size=batch_size)
    elif dataset == 'CIFAR10':
        datamodule = CIFAR10DataModule(domain=domain, batch_size=batch_size)
    elif dataset == 'ImageNet':
        datamodule = ImageNetDataModule(domain=domain, batch_size=batch_size)
    return datamodule


if __name__ == '__main__':
    args = get_args()

    pruner = (
        optuna.pruners.MedianPruner() if args.pruning else optuna.pruners.NopPruner()
    )

    study = optuna.create_study(direction='maximize', pruner=pruner)
    study.optimize(
        lambda trial: objective(
            trial,
            build_model(
                trial, args.architecture, args.domain, dataset_info_table[args.dataset]
            ),
            build_datamodule(args.dataset, args.domain, args.batchsize),
            args.epochs,
            name = f'lightning_logs/{args.architecture}_{args.domain}'
        ),
        n_trials=3,
        timeout=3600,
    )

    print('Number of finished trials: {}'.format(len(study.trials)))

    print('Best trial:')
    trial = study.best_trial

    print('  Value: {}'.format(trial.value))

    print('  Params: ')
    for key, value in trial.params.items():
        print('    {}: {}'.format(key, value))