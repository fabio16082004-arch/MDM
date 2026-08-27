
import os
import random

import numpy as np
import torch
from torch.utils.data import Dataset
import sys
import os

sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'tools'
))
from dataset_index import CONTACT_SLICE
#from data_loaders.slerp_utilities import build_slerp_tail
import json

ALLOWED_DIRECTIONS = {'forward'}


def load_mean_std(mean_std_dir):
    if not mean_std_dir:
        return None, None
    mean = np.load(os.path.join(mean_std_dir, 'Mean.npy'))
    std = np.load(os.path.join(mean_std_dir, 'Std.npy'))
    return mean, std


def normalize(cut, mean, std):
    if mean is None:
        return cut
    return (cut - mean) / std


def sample_random_crop(rng, start_frame, end_frame, core_start, core_end, min_frames=0):
    safe_len = end_frame - start_frame
    core_len = core_end - core_start + 1
    min_crop_len = max(core_len, min_frames)

    if safe_len <= min_crop_len:
        return core_start, core_end + 1

    crop_len = rng.randint(min_crop_len, safe_len)

    earliest_start = max(start_frame, core_end - crop_len + 1)
    latest_start = min(core_start, end_frame - crop_len)

    if latest_start < earliest_start:
        return core_start, core_end + 1

    crop_start = rng.randint(earliest_start, latest_start)
    return crop_start, crop_start + crop_len


def min_frames_for_class(L, fps=20, min_sec_1=1.0, min_sec_2=1.5, min_sec_multi=2.0):
    L = int(L)
    if L == 1:
        min_sec = min_sec_1
    elif L == 2:
        min_sec = min_sec_2
    else:
        min_sec = min_sec_multi
    return int(min_sec * fps)

class CustomDatasetLoader(Dataset):
    def __init__(self, step_index, hand_written_captions, per_class_count,
                 seed=42, mean_std_dir=None, donors_split_path=None):
        self.hand_written_captions = hand_written_captions
        self.per_class_count = per_class_count
        self.mean, self.std = load_mean_std(mean_std_dir)

        if donors_split_path:
            with open(donors_split_path, encoding='utf-8') as f:
                allowed_donors = {line.strip() for line in f if line.strip()}
            step_index = {
                L: [e for e in entries if e['donor_path'] in allowed_donors]
                for L, entries in step_index.items()
            }

        self.step_index = {}
        for L, entries in step_index.items():
            filtered = [e for e in entries if e.get('direction') in ALLOWED_DIRECTIONS]
            self.step_index[str(L)] = filtered
            n_discarded = len(entries) - len(filtered)
            if n_discarded:
                print(f"[CustomDatasetLoader] class {L}: discarded {n_discarded} "
                      f"non-forward windows (backward/lateral/stationary), "
                      f"{len(filtered)} remaining")

        for L, count in per_class_count.items():
            available = self.step_index.get(str(L), [])
            if len(available) == 0:
                raise ValueError(
                    f"No windows available for class {L} in step_index "
                    f"after the lateral/None filter (requested {count} examples "
                    f"per epoch, but the pool is empty)."
                )

        rng = random.Random(seed)
        self.idx_to_class = []
        for L, count in per_class_count.items():
            self.idx_to_class += [L] * count
        rng.shuffle(self.idx_to_class)

    def __len__(self):
        return len(self.idx_to_class)

    def inv_transform(self, sample):
        if self.mean is None:
            return sample
        mean = torch.from_numpy(self.mean).to(sample.device).float()
        std = torch.from_numpy(self.std).to(sample.device).float()
        return sample * std + mean

    def __getitem__(self, idx):
        target_L = self.idx_to_class[idx]
        candidates = self.step_index[str(target_L)]
        choice = random.choice(candidates)

        data = np.load(choice['donor_path'])

        #min_frames = min_frames_for_class(target_L)

        '''crop_start, crop_end = sample_random_crop(
            random, choice['start_frame'], choice['end_frame'],
            choice['core_start'], choice['core_end'],
            min_frames=min_frames
        )'''
        crop_start, crop_end = choice['core_start'], choice['core_end'] + 1
        cut = data[crop_start:crop_end]

        cut = normalize(cut, self.mean, self.std)

        caption = random.choice(self.hand_written_captions[target_L])

        return torch.from_numpy(cut).float(), caption



class FixedEvalDatasetLoader(Dataset):
    def __init__(self, step_index, hand_written_captions, per_class_count,
                 seed=123, mean_std_dir=None, donors_split_path=None):
        self.mean, self.std = load_mean_std(mean_std_dir)
        rng = random.Random(seed)

        if donors_split_path:
            with open(donors_split_path, encoding='utf-8') as f:
                allowed_donors = {line.strip() for line in f if line.strip()}
            step_index = {
                L: [e for e in entries if e['donor_path'] in allowed_donors]
                for L, entries in step_index.items()
            }

        filtered = {}
        for L, entries in step_index.items():
            filtered[str(L)] = [e for e in entries if e.get('direction') in ALLOWED_DIRECTIONS]

        self.fixed_items = []
        for L, count in per_class_count.items():
            available = filtered.get(str(L), [])
            if len(available) == 0:
                raise ValueError(
                    f"No windows available for class {L} in step_index "
                    f"(val/test) after the lateral/None filter."
                )
            if len(available) >= count:
                chosen = rng.sample(available, count)   
            else:
                print(f"[FixedEvalDatasetLoader] class {L}: pool of {len(available)} "
                      f"windows insufficient for {count} requested - some "
                      f"will be repeated (with replacement).")
                chosen = [rng.choice(available) for _ in range(count)]

            min_frames = min_frames_for_class(L)

            for entry in chosen:
                caption = rng.choice(hand_written_captions[L])  

                crop_start, crop_end = sample_random_crop(
                    rng, entry['start_frame'], entry['end_frame'],
                    entry['core_start'], entry['core_end'],
                    min_frames=min_frames
                )

                self.fixed_items.append({
                    'donor_path': entry['donor_path'],
                    'start_frame': crop_start,
                    'end_frame': crop_end,
                    'caption': caption,
                    'L': L,
                })

        rng.shuffle(self.fixed_items)

    def __len__(self):
        return len(self.fixed_items)

    def inv_transform(self, sample):
        if self.mean is None:
            return sample
        mean = torch.from_numpy(self.mean).to(sample.device).float()
        std = torch.from_numpy(self.std).to(sample.device).float()
        return sample * std + mean

    def __getitem__(self, idx):
        item = self.fixed_items[idx]
        data = np.load(item['donor_path'])
        cut = data[item['start_frame']:item['end_frame']]
        cut = normalize(cut, self.mean, self.std)
        return torch.from_numpy(cut).float(), item['caption']