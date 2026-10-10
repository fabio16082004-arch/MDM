# Step-count-conditioned motion generation with MDM

This repository extends MDM to generate HumanML3D walking sequences conditioned
on the number of steps. The custom dataset identifies walking windows with 1, 2,
3, or 4 steps. During training and generation, sequence lengths can be
standardized using tail SLERP (`tail`), head-and-tail SLERP (`head_tail`), or no
SLERP (`none`). The sections below document this project's pipeline; setup of the
original MDM code is covered in [Based on MDM](#based-on-mdm).

### Requirements and dependencies

- The MDM Python/PyTorch/CUDA environment described in
  [`environment.yml`](environment.yml), the HumanML3D dataset, and its
  normalization files `Mean.npy` and `Std.npy` in `dataset/HumanML3D/`.
- The HumanML3D checkpoint `humanml_trans_enc_512/model000475000.pt`.
  The pretrained checkpoint is linked in the [upstream
  instructions](https://github.com/GuyTevet/motion-diffusion-model#3-download-the-pretrained-models). Place the model
  files in `save/humanml_trans_enc_512/`.
- [TMR](https://github.com/Mathux/TMR) (Petrovich et al., ICCV 2023) is used as the
  text-motion embedding model for the TMR matching score and FID. It is a separate
  checkout in `~/TMRPlusPlus`, with the pretrained model at
  `models/tmr_humanml3d_guoh3dfeats` (HumanML3D, Guo 263-dim features). The metric scripts import TMR from this
  location; otherwise, update `TMR_ROOT` in those scripts.
- The Guo evaluator ([text-to-motion](https://github.com/EricGuo5513/text-to-motion),
  Guo et al., CVPR 2022) is distributed/managed separately from MDM. The
  `count_with_tolerance.py` and `count_without_cutoff.py` scripts look for the
  bundle in `guo_evaluator_bundle/` at the repository root. If installed
  elsewhere, make it available at that path (for example, with a symlink). The
  Guo token cache required by the metrics is `assets/guo_caption_tokens.json`.

Activate the MDM environment and run commands from the repository root. For
evaluation, also activate an environment compatible with TMR and the Guo
bundle. See [`guo_evaluator_bundle/README.md`](guo_evaluator_bundle/README.md)
for bundle installation details and input requirements.

### 1. Prepare the splits

The index contains valid windows for each class and references to donor files.
To regenerate the index and splits:

```bash
python scripts/build_step_index.py \
  --donors_list scripts/donors_clean.txt \
  --output_index splits/step_index.json
python rebalance_splits.py
```

`rebalance_splits.py` uses seed 42 and aims to balance forward examples across
classes. The splits currently present in this checkout contain 404/50/52 donors
(train/val/test). Recheck the counts after regenerating the files: they depend
on the index and donor list used.

The examples below use `splits/step_index.json`, which is present in this
checkout. If you have a dedicated filtered index, such as
`splits/step_index_test_for_slerp.json`, set `STEP_INDEX` to that path and use
the same index consistently for training, generation, and evaluation. That
dedicated file is not included in this checkout.

### 2. Prepare the pretrained checkpoint and fine-tune

The `dataset` saved in a checkpoint's `args.json` may override the `--dataset`
argument. To create a copy with the dataset set to the custom dataset for
generation:

```bash
mkdir -p save/humanml_trans_enc_512_custom
cp save/humanml_trans_enc_512/model000475000.pt \
  save/humanml_trans_enc_512_custom/
python -c "import json; p='save/humanml_trans_enc_512/args.json'; a=json.load(open(p)); a['dataset']='custom_dataset'; json.dump(a, open('save/humanml_trans_enc_512_custom/args.json','w'), indent=2)"
```

Fine-tuning loads weights from the original checkpoint. Use a new output
directory for each experiment; by default, MDM will not overwrite an existing
training directory.

```bash
STEP_INDEX=splits/step_index.json

# Tail SLERP
python -m train.train_mdm --dataset custom_dataset \
  --slerp_mode tail \
  --save_dir save/tail \
  --fine_tune_path save/humanml_trans_enc_512/model000475000.pt \
  --step_index "$STEP_INDEX" \
  --lr 1e-5 --save_interval 1000 --num_steps 35000 \
  --use_ema --seed 42 --device 0

# No SLERP
python -m train.train_mdm --dataset custom_dataset \
  --slerp_mode none --mask_frames \
  --save_dir save/none \
  --fine_tune_path save/humanml_trans_enc_512/model000475000.pt \
  --step_index "$STEP_INDEX" \
  --lr 1e-5 --save_interval 1000 --num_steps 35000 \
  --use_ema --seed 42 --device 0

# Head-and-tail SLERP
python -m train.train_mdm --dataset custom_dataset \
  --slerp_mode head_tail \
  --save_dir save/head_tail \
  --fine_tune_path save/humanml_trans_enc_512/model000475000.pt \
  --step_index "$STEP_INDEX" \
  --lr 1e-5 --save_interval 1000 --num_steps 35000 \
  --use_ema --seed 42 --device 0
```

### 3. Generate samples on the test set

The SLERP mode and EMA setting are read from the arguments saved with the
checkpoint. Three repetitions generate three sequences for each item in the
split (in the experiment described here, 100 items produce 300 samples).

```bash
STEP_INDEX=splits/step_index.json
python -m sample.generate \
  --model_path save/tail/model000035000.pt \
  --dataset custom_dataset --step_index "$STEP_INDEX" --split test \
  --num_repetitions 3 --skip_rendering --seed 42 --device 0 \
  --output_dir save/tail/samples_test_35000
```

For the other experiments, replace the checkpoint and output directory with
the corresponding paths.
`results.npy` is saved in a directory that also includes the split name (for
example, `samples_test_35000_split-test/results.npy`).

### 4. Metrics and confusion matrix

The step-count metrics compare the requested and detected step counts and also
report TMR matching/FID and Guo matching/FID. TMR and Guo require the respective
models and external resources described above.

```bash
STEP_INDEX=splits/step_index.json
RESULTS=save/tail/samples_test_35000_split-test/results.npy

# Accuracy, +/-1-step tolerance, TMR, and Guo metrics;
# uses the script's default cutoff settings
python scripts/count_with_tolerance.py \
  --results_npy "$RESULTS" --step_index "$STEP_INDEX" --split test


# Count over the entire sequence, without a cutoff
python scripts/count_without_cutoff.py \
  --results_npy "$RESULTS" --step_index "$STEP_INDEX" --split test
```

### 5. Render generated samples alongside ground truth

The renderer places generated samples alongside ground-truth sequences from
the same split. `--slerp_mode` must match the configuration used for the
checkpoint.

```bash
python scripts/render_comparison.py \
  --results_npy save/tail/samples_test_35000_split-test/results.npy \
  --baseline_results_npy save/tail/samples_test_0_split-test/results.npy \
  --step_index splits/step_index.json --split test \
  --outdir render_comparison_tail --n_examples 10 --slerp_mode tail
```

`baseline_results_npy` must point to valid generated results for the same
split; replace it with the path to your baseline results if different. The
renderer saves comparison videos in the output directory.

---

---

## Citation

If you use this code, please cite MDM, and TMR and the Guo evaluator if you use the respective metrics:

```
@inproceedings{
tevet2023human,
title={Human Motion Diffusion Model},
author={Guy Tevet and Sigal Raab and Brian Gordon and Yoni Shafir and Daniel Cohen-or and Amit Haim Bermano},
booktitle={The Eleventh International Conference on Learning Representations },
year={2023},
url={https://openreview.net/forum?id=SJ1kSyO2jwu}
}

@inproceedings{petrovich23tmr,
  title     = {{TMR}: Text-to-Motion Retrieval Using Contrastive {3D} Human Motion Synthesis},
  author    = {Petrovich, Mathis and Black, Michael J. and Varol, G{\"u}l},
  booktitle = {International Conference on Computer Vision ({ICCV})},
  year      = {2023}
}

@inproceedings{guo2022generating,
  title     = {Generating Diverse and Natural 3D Human Motions from Text},
  author    = {Guo, Chuan and Zou, Shihao and Zuo, Xinxin and Wang, Sen and Ji, Wei and Li, Xingyu and Cheng, Li},
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  pages     = {5152--5161},
  year      = {2022}
}
```