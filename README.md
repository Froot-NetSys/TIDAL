# Tidal: Tackling Concept Drift in Provenance-based Advanced Persistent Threats Detection

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

### 2. Merge Sketch Data

For reproducing the released results, the GraphChi sketch vectors are already
merged into `train/data/<dataset>/*.json` as `z`. If you run TIDAL on your own
dataset, generate the sketch files with the modified GraphChi implementation
from https://github.com/crimson-unicorn, then merge them with the processed
logs. The `sketch_toy.zip` files are the precomputed `sketch-toy-*.txt`
GraphChi sketch outputs:

```bash
cd graph_data_process
python map_sketch_raw_data.py
```

### 3. Split Train and Test

For reproduction, the released train/test data are already provided as JSON
files under `train/data/<dataset>/`.

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
5. **Graph Sketch Fusion** — optional concatenation of GraphChi sketch vectors
6. **Classifier Head** — binary output (benign vs. attack)

## Data Format

**Raw log format** (tab-separated):

```
srcUUID    dstUUID    action    target    timestamp
```

**Training JSON** (`[x, y, z]`):

- `x` — list of token sequences (list of int lists)
- `y` — binary labels (`0` = benign, `1` = attack)
- `z` — GraphChi sketch vector per sequence
