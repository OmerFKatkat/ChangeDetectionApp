import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import utils
import data_config
from datasets.CD_dataset import CDDataset

def get_loader(data_name, img_size=256, batch_size=8, split='test',
               is_train=False, dataset='CDDataset'):
    dataConfig = data_config.DataConfig().get_data_config(data_name)
    root_dir = dataConfig.root_dir
    label_transform = dataConfig.label_transform
    if dataset == 'CDDataset':
        data_set = CDDataset(root_dir=root_dir, split=split,
                             img_size=img_size, is_train=is_train,
                             label_transform=label_transform)
    else:
        raise NotImplementedError(
            'Wrong dataset name %s (choose one from [CDDataset])' % dataset)
    dataloader = DataLoader(data_set, batch_size=batch_size,
                            shuffle=is_train, num_workers=4)
    return dataloader

def get_loaders(args):
    data_name = args.data_name
    dataConfig = data_config.DataConfig().get_data_config(data_name)

    root_dir = dataConfig.root_dir
    label_transform = dataConfig.label_transform

    split = args.split
    split_val = getattr(args, 'split_val', 'val')

    print("========== Dataset Info ==========")
    print("data_name:", data_name)
    print("root_dir:", root_dir)
    print("train split:", split)
    print("val split:", split_val)
    print("label_transform:", label_transform)
    print("==================================")

    if args.dataset == 'CDDataset':
        training_set = CDDataset(
            root_dir=root_dir,
            split=split,
            img_size=args.img_size,
            is_train=True,
            label_transform=label_transform
        )

        val_set = CDDataset(
            root_dir=root_dir,
            split=split_val,
            img_size=args.img_size,
            is_train=False,
            label_transform=label_transform
        )
    else:
        raise NotImplementedError(
            'Wrong dataset name %s (choose one from [CDDataset])'
            % args.dataset
        )

    dataloaders = {
        'train': DataLoader(
            training_set,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers
        ),
        'val': DataLoader(
            val_set,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers
        )
    }

    print("Train size:", len(training_set))
    print("Val size:", len(val_set))

    return dataloaders

def make_numpy_grid(tensor_data, pad_value=0, padding=0):
    tensor_data = tensor_data.detach()
    vis = utils.make_grid(tensor_data, pad_value=pad_value, padding=padding)
    vis = np.array(vis.cpu()).transpose((1, 2, 0))
    if vis.shape[2] == 1:
        vis = np.stack([vis, vis, vis], axis=-1)
    return vis

def de_norm(tensor_data):
    return tensor_data * 0.5 + 0.5

def get_device(args):
    str_ids = args.gpu_ids.split(',')
    args.gpu_ids = []
    for str_id in str_ids:
        id = int(str_id)
        if id >= 0:
            args.gpu_ids.append(id)
    if len(args.gpu_ids) > 0:
        torch.cuda.set_device(args.gpu_ids[0])