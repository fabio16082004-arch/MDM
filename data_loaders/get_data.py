from torch.utils.data import DataLoader, Dataset
from data_loaders.tensors import collate as all_collate
from data_loaders.tensors import t2m_collate, t2m_prefix_collate, custom_dataset_collate
from data_loaders.custom_dataloaders import CustomDatasetLoader, FixedEvalDatasetLoader
import os
import numpy as np
import torch
import glob
import re
import random

CUSTOM_TRAIN_PER_CLASS_COUNT = {1: 200, 2: 200, 3: 200, 4: 200}
CUSTOM_EVAL_PER_CLASS_COUNT = {1: 25, 2: 25, 3: 25, 4: 25}

CUSTOM_SPLITS_DIR = 'splits'


def get_dataset_class(name):
    if name == "amass":
        from .amass import AMASS
        return AMASS
    elif name == "uestc":
        from .a2m.uestc import UESTC
        return UESTC
    elif name == "humanact12":
        from .a2m.humanact12poses import HumanAct12Poses
        return HumanAct12Poses
    elif name == "humanml":
        from data_loaders.humanml.data.dataset import HumanML3D
        return HumanML3D
    elif name == "kit":
        from data_loaders.humanml.data.dataset import KIT
        return KIT
    elif name == "custom_dataset":
        return CustomDatasetLoader
    else:
        raise ValueError(f'Unsupported dataset name [{name}]')

def get_collate_fn(name, hml_mode='train', pred_len=0, batch_size=1, num_frames=None):
    if hml_mode == 'gt':
        from data_loaders.humanml.data.dataset import collate_fn as t2m_eval_collate
        return t2m_eval_collate
    if name == "custom_dataset":
        return lambda x: custom_dataset_collate(x, target_len=num_frames)
    if name in ["humanml", "kit"]:
        if pred_len > 0:
            return lambda x: t2m_prefix_collate(x, pred_len=pred_len)
        return lambda x: t2m_collate(x, batch_size)
    else:
        return all_collate


def get_dataset(name, num_frames, split='train', hml_mode='train', abs_path='.', fixed_len=0, 
                device=None, autoregressive=False, cache_path=None, custom_data_dir = "",
                step_index_path=None, mean_std_dir=None, slerp_mode='tail'):
    if name in ["humanml", "kit"]:
        DATA = get_dataset_class(name)
        dataset = DATA(split=split, num_frames=num_frames, mode=hml_mode, abs_path=abs_path, fixed_len=fixed_len, 
                       device=device, autoregressive=autoregressive)
    elif name in ["amass", "uestc", "humanact12"]:
        DATA = get_dataset_class(name)
        dataset = DATA(split=split, num_frames=num_frames)
    elif name == 'custom_dataset':
        import json
        from captions import build_hand_written_captions

        with open(step_index_path) as f:
            step_index = json.load(f)
        hand_written_captions = build_hand_written_captions()

        donors_split_path = os.path.join(CUSTOM_SPLITS_DIR, f'{split}.txt')

        if split == 'train':
            dataset = CustomDatasetLoader(step_index, hand_written_captions,
                                          CUSTOM_TRAIN_PER_CLASS_COUNT, seed=42,
                                          mean_std_dir=mean_std_dir,
                                          donors_split_path=donors_split_path,
                                          slerp_mode=slerp_mode)
        else:  # val/test
            dataset = FixedEvalDatasetLoader(step_index, hand_written_captions,
                                             CUSTOM_EVAL_PER_CLASS_COUNT, seed=123,
                                             mean_std_dir=mean_std_dir,
                                             donors_split_path=donors_split_path,
                                             slerp_mode=slerp_mode)
    else:
        raise ValueError(f'Unsupported dataset name [{name}]')
    return dataset


def get_dataset_loader(name, batch_size, num_frames, split='train', hml_mode='train', fixed_len=0, pred_len=0, 
                       device=None, autoregressive=False, custom_data_dir="",
                       step_index_path=None, mean_std_dir=None, shuffle=True,
                       slerp_mode='tail'):
    dataset = get_dataset(name, num_frames, split=split, hml_mode=hml_mode, fixed_len=fixed_len, 
                device=device, autoregressive=autoregressive, custom_data_dir=custom_data_dir,
                step_index_path=step_index_path, mean_std_dir=mean_std_dir,
                slerp_mode=slerp_mode)
    
    collate = get_collate_fn(name, hml_mode, pred_len, batch_size, num_frames=num_frames)

    n_samples = len(dataset)
    if n_samples == 0:
        raise ValueError(
            f"Dataset '{name}' split='{split}' is empty (0 files found). "
            f"Please check custom_data_dir='{custom_data_dir}' and the structure of the inner directories"
        )
    if n_samples < batch_size:
        print(f"[WARNING] split='{split}': {n_samples} samples < batch_size={batch_size}. "
              f"With drop_last=True this split will produce ZERO batch per epoch.")


    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle,
        num_workers=8, drop_last=(split == 'train'), collate_fn=collate
    )

    return loader