# APT-Transformer

A transformer-based framework for detecting Advanced Persistent Threats (APTs) in system audit logs. The pipeline processes raw DARPA TC3 provenance data, builds semantic token trees from file paths, and trains a transformer encoder to classify log sequences as benign or attack.

## Pipeline Overview

```
Raw DARPA TC3 Logs
        │
        ▼
 1. darpa_tc3_convert        ──  Parse raw audit logs into structured format
        │
        ▼
 2. graph_data_process/      ──  Merge GraphChi sketch vectors with logs
        │
        ▼
 3. sample_train_test        ──  Split data into train/test sets
        │
        ▼
 4. generate_token_tree/     ──  Build hierarchical token tree from file paths
        │
        ▼
 5. tokenize_train_test/     ──  Tokenize logs and resample for class balance
        │
        ▼
 6. train/                   ──  Train and evaluate the transformer classifier
```

## Project Structure

```
├── generate_token_tree/       # Semantic tokenization tree construction
│   ├── nlp_stem.py            # Extract unique actions/objects from logs via NLTK
│   ├── tree_from_path.py      # Build hierarchical tree from file paths
│   └── object_data/           # Per-dataset action and object vocabularies
│
├── graph_data_process/        # Sketch data integration
│   └── map_sketch_raw_data.py # Merge GraphChi sketch vectors with raw logs
│
├── tokenize_train_test/       # Log tokenization and data preparation
│   ├── token_and_resample.py  # Tokenize, chunk sequences, resample train/test
│   ├── semi_token_and_resample.py  # Semi-supervised variant with pseudo-labels
│   └── fast_token_utils.py    # Core tokenization utilities (tree lookup, hashing)
│
└── train/                     # Model training and evaluation
    ├── train.py               # Training loop (AdamW, self-training, F1 selection)
    ├── config.py              # Dataset paths and hyperparameters
    ├── ClassificationModel.py # Token/positional embedding → transformer → classifier
    ├── MyTransformer.py       # Custom transformer encoder with multi-head attention
    ├── Embedding.py           # Positional encoding and token embedding layers
    ├── data_helper.py         # JSON data loading and DataLoader construction
    ├── test_log.py            # Threshold sweep evaluation, outputs CSV
    └── test_roc.py            # ROC and precision-recall curve generation
```

## Requirements

- Python 3.8+
- PyTorch >= 1.10.0 (CUDA recommended)
- pandas
- numpy
- tqdm
- nltk
- scikit-learn
- matplotlib

Install all dependencies:

```bash
pip install torch pandas numpy tqdm nltk scikit-learn matplotlib
```

## Usage

### 1. Process Raw DARPA Data

Convert raw DARPA TC3 audit logs into the structured format (`srcUUID  dstUUID  action  target  timestamp`):

```bash
# Use the darpa_tc3_convert tool (external)
```

### 2. Merge Sketch Data

Combine GraphChi sketch vectors with the processed logs:

```bash
cd graph_data_process
python map_sketch_raw_data.py
```

### 3. Split Train and Test

```bash
# Use the sample_train_test tool (external)
```

### 4. Generate Token Tree

Extract unique objects/actions and build the semantic token tree:

```bash
cd generate_token_tree
python nlp_stem.py > object_data/cadet_object.txt
python tree_from_path.py > object_data/cadet_tree.txt
```

### 5. Tokenize Logs

Tokenize the log data and resample for class balance:

```bash
cd tokenize_train_test
python token_and_resample.py --semi_data_path='../data/cadet/full_data/graph/graph_data_sampled/'
```

For semi-supervised learning with pseudo-labels:

```bash
python semi_token_and_resample.py --semi_data_path='../data/cadet/full_data/graph/graph_data_sampled/'
```

### 6. Train the Model

Configure the dataset and model in `train/config.py`:

- `dataset_dir` — path to the dataset
- `train_corpus_file_paths` / `test_corpus_file_paths` — JSON filenames
- `vocab_size` — set to match the token tree size
- `model_name` — name for saved checkpoints

Then train:

```bash
cd train
python train.py
```

### 7. Evaluate

Run threshold-based evaluation and generate metrics:

```bash
python test_log.py      # Threshold sweep, outputs results CSV
python test_roc.py      # ROC and precision-recall curves
```

## Model Architecture

The classifier follows an encoder-only transformer design:

1. **Token Embedding** — maps tokenized log events to dense vectors
2. **Positional Encoding** — injects sequence position information
3. **Transformer Encoder** — multi-head self-attention over the log sequence
4. **Pooling** — sum/average over sequence length
5. **Graph Embedding Fusion** — optional concatenation of node2vec embeddings
6. **Classifier Head** — binary output (benign vs. attack)

## Supported Datasets

| Dataset     | Source     |
|-------------|------------|
| CADETS      | DARPA TC3  |
| TRACE       | DARPA TC3  |
| THEIA       | DARPA TC3  |
| FiveD       | DARPA TC3  |
| ClearScope  | DARPA TC3  |

## Data Format

**Raw log format** (tab-separated):

```
srcUUID    dstUUID    action    target    timestamp
```

**Training JSON** (`[x, y, z]`):

- `x` — list of token sequences (list of int lists)
- `y` — binary labels (`0` = benign, `1` = attack)
- `z` — node2vec graph embeddings per sequence
