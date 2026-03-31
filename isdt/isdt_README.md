# ISDT Pipeline

This repository provides the full pipeline for **ISDT pretraining, prototype-based reasoning, token construction, and statistical analysis** on brain graph datasets (e.g., ABIDE).

---

## 1. Pretraining ISDT

Train the ISDT model on the original brain graph dataset.

### Command

```bash
python isdt/main.py \
  --dataset ABIDE \
  --aal_xlsx ./isdt/AAL.xlsx \
  --epochs 80 \
  --batch_size 8 \
  --topk 10 \
  --save_dir ./isdt/isdt_ckpt_abide \
  --save_best
```

### Output

```bash
isdt/isdt_ckpt_abide/
 ├── isdt_epoch_*.pt
 └── isdt_best.pt
```

## 2. Token Reasoning (Prototype Inference)

Use the pretrained ISDT model to perform prototype reasoning and export node-level tokens.

### Command

```bash
python isdt/export_token.py \
  --dataset ABIDE \
  --topk 10 \
  --batch_size 8 \
  --aal_xlsx ./isdt/AAL.xlsx \
  --save_dir ./isdt/isdt_ckpt_abide \
  --argmax_code \
  --max_graphs -1
```

### output:

```bash
isdt/isdt_ckpt_abide/
 ├── tokens_train.pt   # list of [116,3]
 ├── tokens_dev.pt
 └── tokens_test.pt
```

## 3. Build Token-level Dataset

Convert node-level tokens into fixed-length token sequences for downstream models (e.g., LLMs).

### Command

```bash
python isdt/build_token_dataset.py \
  --save_dir ./isdt/isdt_ckpt_abide \
  --top_m 32 \
  --K 256 \
  --make_single_id
```

### output

```bash
isdt/isdt_ckpt_abide/
 ├── token_dataset_train.pt
 ├── token_dataset_dev.pt
 └── token_dataset_test.pt
```

## 4. Statistics and Analysis

Perform statistical inspection and qualitative analysis of prototypes and token distributions.

### Command

python isdt/print.py
