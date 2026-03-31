# BrainMoE

BrainMoE is a brain graph learning framework for psychiatric disorder analysis.  
It combines identity-aware structural representations with **LLM-guided** mixture-of-experts routing to support interpretable graph learning on brain network data.

Currently, this repository supports experiments on:

- **ADHD**
- **ABIDE**

---

## Setup

We recommend using Conda.

```bash
conda create -n brainmoe python=3.12
conda activate brainmoe

pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cu129
pip install torch-geometric==2.7.0
pip install --no-index torch_sparse==0.6.18 torch_scatter==2.1.2 torch_cluster==1.6.3 torch_spline_conv==1.2.2 -f https://data.pyg.org/whl/torch-2.8.0%2Bcu129.html
pip install -r requirements.txt
```

- Change your dataset path in `src/utils.py`.
- If you want to generate LLM output, change `client = genai.Client(api_key="API_KEY")` in `llm/llm_utils.py`.

## LLM Cache Generation

BrainMoE supports generating subject-level LLM reasoning outputs and text embeddings in advance, so that downstream training can directly load cached results instead of repeatedly querying the LLM.

The generated cache is typically stored under:

```bash
data/abnormal_llm_cache/<DATASET>/
```

These cached files may include Stage-A / Stage-B LLM outputs, text embeddings, and packaged artifacts for later reuse.

### Example: Generate LLM Cache

```bash
python list_models.py \
  --dataset ADHD \
  --cached_pt data/preprocessed_subjects/ADHD_cached.pt \
  --aal_xlsx isdt/AAL.xlsx \
  --n_splits 5 \
  --seed 0 \
  --val_ratio 0.1 \
  --llm_workers 1 \
  --emb_workers 2 \
  --topk_edges 20 \
  --topk_rois 10 \
  --pack_tar
```

### Debug Mode
```bash
python list_models.py \
  --dataset ADHD \
  --cached_pt data/preprocessed_subjects/ADHD_cached.pt \
  --aal_xlsx isdt/AAL.xlsx \
  --n_splits 5 \
  --seed 0 \
  --val_ratio 0.1 \
  --llm_workers 1 \
  --emb_workers 2 \
  --topk_edges 20 \
  --topk_rois 10 \
  --debug_limit 5 \
  --pack_tar
```

### ISDT Ablation Example
```bash
python build_global_llm_cache_ablate_isdt.py \
  --dataset ADHD \
  --cached_pt data/preprocessed_subjects/ADHD_cached.pt \
  --aal_xlsx isdt/AAL.xlsx \
  --global_mode \
  --ablate_isdt all \
  --max_samples 20
```

### Generate global LLM cache with ISDT ablation

This script generates global abnormality-driven LLM outputs with ISDT ablated.

```bash
python list_models.py `
  --dataset ADHD `
  --cached_pt "D:\code\BrainMoE-02\data\preprocessed_subjects\ADHD_cached.pt" `
  --aal_xlsx "D:\code\BrainMoE-02\isdt\AAL.xlsx" `
  --n_splits 5 --seed 0 --val_ratio 0.1 `
  --llm_workers 1 --emb_workers 2 `
  --topk_edges 20 --topk_rois 10 `
  --pack_tar
```

### How the LLM data is used

The overall workflow is:

preprocess subject data into *_cached.pt
generate LLM reasoning outputs and embeddings into data/abnormal_llm_cache/...
run main.py
during training, BrainMoE loads the cached LLM outputs and embeddings instead of re-calling the LLM

In other words, the LLM data is produced offline first, then consumed during model training.

## Run


### Quick Start
```bash
python main.py
```
### Common Options

```bash
python main.py --dataset ADHD
python main.py --dataset ABIDE
python main.py --num_epochs 40 --batch_size 32
```

### Main Training Commands

Final ADHD configuration:

```bash
python main.py \
  --dataset ADHD \
  --use_identity \
  --use_llm_stage1 \
  --use_llm_stage2 \
  --llm_hidden_dim 64 \
  --router_noise_std 0.055 \
  --lambda_prior 0.0 \
  --lambda_soft_graph_balance 0.003 \
  --lambda_soft_expert_balance 0.003 \
  --lambda_min_usage 0.0 \
  --lambda_importance 0.01 \
  --lambda_load 0.01 \
  --lambda_z 1e-4 \
  --num_epochs 60 \
  --early_stop_patience 12 \
  --selection_metric accuracy \
  --threshold_metric accuracy \
  --lr 5e-4
```
Final 5-fold summary on ADHD:

```bash
balanced_acc : 0.5951 ± 0.0178
acc          : 0.6588 ± 0.0266
auc          : 0.6594 ± 0.0340
f1           : 0.4255 ± 0.0751
precision    : 0.5738 ± 0.0714
recall       : 0.3607 ± 0.1220
```

Final ABIDE configuration:

```bash
python main.py \
  --dataset ABIDE \
  --seed 42 \
  --hidden_dim 128 \
  --llm_hidden_dim 64 \
  --top_k 2 \
  --dropout 0.1 \
  --batch_size 32 \
  --num_epochs 100 \
  --lr 5e-4 \
  --weight_decay 1e-4 \
  --lambda_prior 0.005 \
  --lambda_soft_graph_balance 0.0005 \
  --lambda_soft_expert_balance 0.0005 \
  --lambda_importance 0.002 \
  --lambda_load 0.002 \
  --lambda_z 1e-4 \
  --router_noise_std 0.03 \
  --router_temperature 2.5 \
  --stage1_scale_init 0.35 \
  --early_stop_patience 24 \
  --n_splits 5 \
  --val_ratio 0.1 \
  --use_class_weight \
  --selection_metric accuracy \
  --threshold_metric accuracy \
  --use_identity \
  --use_llm_stage1 \
  --use_llm_stage2 \
  --results_dir results_brainmoe_ABIDE_llm_both_lr5e4
```
Final 5-fold summary on ABIDE:

```bash
balanced_acc : 0.6502 ± 0.0134
acc          : 0.6502 ± 0.0139
auc          : 0.7082 ± 0.0319
f1           : 0.6546 ± 0.0295
precision    : 0.6624 ± 0.0195
recall       : 0.6509 ± 0.0667
```

### Routing Analysis and Visualization

This repository includes scripts for subject-level routing extraction, stability analysis, and visualization of network–expert relationships.

#### Step 1: Build subject-level routing CSV
ADHD
```bash
python build_routing_subject_level_from_accuracy_node_pt.py \
  --results_dir results_brainmoe_identity \
  --dataset ADHD \
  --aal_xlsx isdt/AAL.xlsx \
  --out_csv results_brainmoe_identity/network_expert_arrays/routing_subject_level_accuracy_soft_with_fold_ADHD.csv \
  --use_soft
```

ABIDE
```bash
python build_routing_subject_level_from_accuracy_node_pt.py \
  --results_dir results_brainmoe_ABIDE_identity_stage1_stage2 \
  --dataset ABIDE \
  --aal_xlsx isdt/AAL.xlsx \
  --out_csv results_brainmoe_ABIDE_identity_stage1_stage2/network_expert_arrays/routing_subject_level_accuracy_soft_with_fold_ABIDE.csv \
  --use_soft
```


#### Step 2: Analyze routing stability
ADHD
```bash
python analyze_routing_stability.py \
  --csv results_brainmoe_identity/network_expert_arrays/routing_subject_level_accuracy_soft_with_fold_ADHD.csv \
  --outdir results_brainmoe_identity/routing_stability_accuracy_soft \
  --networks SMN,DMN,FPN,DAN,VN,LIN,SBN,VAN,CBL \
  --experts mlp,cheb,gt,gcn \
  --n_boot 2000 \
  --top_k 10
```

ABIDE
```bash
python analyze_routing_stability.py \
  --csv results_brainmoe_ABIDE_identity_stage1_stage2/network_expert_arrays/routing_subject_level_accuracy_soft_with_fold_ABIDE.csv \
  --outdir results_brainmoe_ABIDE_identity_stage1_stage2/routing_stability_accuracy_soft \
  --networks SMN,DMN,FPN,DAN,VN,LIN,SBN,VAN,CBL \
  --experts mlp,cheb,gt,gcn \
  --n_boot 2000 \
  --top_k 10
```

#### Step 3: Plot routing manifold and heatmaps
ADHD
```bash
python plot_neurips_routing_manifold.py \
  --result_dir results_brainmoe_identity/routing_stability_accuracy_soft \
  --save_dir results_brainmoe_identity/routing_figures_neurips_clean \
  --networks SMN,DMN,FPN,DAN,VN,LIN,SBN,VAN,CBL \
  --experts mlp,cheb,gt,gcn \
  --fontsize 13 \
  --annotate_heatmap
```

ABIDE
```bash
python plot_neurips_routing_manifold.py \
  --result_dir results_brainmoe_ABIDE_identity_stage1_stage2/routing_stability_accuracy_soft \
  --save_dir results_brainmoe_ABIDE_identity_stage1_stage2/routing_figures_neurips_clean \
  --networks SMN,DMN,FPN,DAN,VN,LIN,SBN,VAN,CBL \
  --experts mlp,cheb,gt,gcn \
  --fontsize 13 \
  --annotate_heatmap
```