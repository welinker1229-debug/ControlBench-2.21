# ControBench: A Comprehensive Benchmark for Controversial Discourse Analysis on Social Networks

This repository contains the implementation and experiments for ControBench, a benchmark dataset that combines heterogeneous graph structures with rich textual semantics for controversial discourse analysis on social networks.

## 📋 Table of Contents

- [Overview](#overview)
- [Installation](#installation)
- [Dataset Structure](#dataset-structure)
- [Quick Start](#quick-start)
- [Experiments](#experiments)
  - [Graph Neural Networks (GNN)](#graph-neural-networks-gnn)
  - [Pre-trained Language Models (PLM)](#pre-trained-language-models-plm)
  - [Large Language Models (LLM)](#large-language-models-llm)
- [Hyperparameter Tuning](#hyperparameter-tuning)
- [File Structure](#file-structure)
- [Citation](#citation)

## 🎯 Overview

ControBench bridges the gap between textual semantics and social interaction context for controversial discourse analysis. The benchmark includes:

- **5 controversial topics**: Trump politics, abortion ethics, LGBTQ rights, capitalism vs. socialism, religious beliefs
- **13,959 users** and **28,265 interactions**
- **Heterogeneous graph structure** with dual semantic edge features
- **Three experimental paradigms**: GNNs, PLMs, and LLMs

## 🔧 Installation

### Prerequisites

- Python 3.8+
- CUDA-compatible GPU (recommended for deep learning experiments)

### Environment Setup

1. **Clone the repository**
```bash
git clone <repository-url>
cd controbench
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Set environment variables**
```bash
cp .env.example .env
# fill in API keys (e.g., OPENROUTER_API_KEY, HF_TOKEN, TOGETHER_API_KEY) in .env
```

4. **Additional dependencies for specific experiments**

For SimCSE experiments:
```bash
pip install simcse
```

For Sentence-BERT experiments:
```bash
pip install sentence-transformers
```

For LLM experiments, you'll need API keys for:
- Together AI (recommended for cost-effective experiments)
- OpenAI (for GPT models)
- Other providers as needed

## 📁 Dataset Structure

The datasets are organized as follows:

```
data/
├── graph_data_abortion.json       # Original graph data
├── graph_data_capitalism.json
├── graph_data_lgbtq.json
├── graph_data_religion.json
├── graph_data_trump.json
└── split_datasets/                # Train/val/test splits
    ├── abortion/
    │   ├── train.json
    │   ├── validation.json
    │   ├── test.json
    │   └── metadata.json
    └── [other datasets...]
```

### Data Preparation

If you need to create train/validation/test splits:

```bash
python split_dataset.py --dataset abortion --output_dir split_datasets
python split_dataset.py --dataset trump --output_dir split_datasets
# ... repeat for other datasets
```

## 🚀 Quick Start

### Basic GNN Experiment
```bash
# Run a basic GNN experiment
python train.py --model HAN --dataset abortion

# Run with hyperparameter tuning
python train.py --model HAN --dataset abortion --hyperparameter_search
```

### Basic PLM Experiment
```bash
# Run text classification experiments with all PLMs
python run_experiments.py abortion trump
```

### Basic LLM Experiment
```bash
# Edit llm_train.py to add your API key, then:
python llm_train.py
```

## 🧪 Experiments

### Graph Neural Networks (GNN)

The GNN experiments support four architectures: RGCN, HAN, HGT, and HinSAGE.

#### Available Models
- **RGCN**: Relational Graph Convolutional Network
- **HAN**: Heterogeneous Graph Attention Network  
- **HGT**: Heterogeneous Graph Transformer
- **HinSAGE**: Heterogeneous GraphSAGE

#### Basic Usage

```bash
# Single model on single dataset
python train.py --model HAN --dataset abortion

# All models on single dataset
python benchmark.py
```

#### Configuration

Models can be configured in `config.py`. Key parameters:
- `hidden_size`: Hidden layer dimensions (default: 256)
- `n_layers`: Number of layers (default: 2)
- `lr`: Learning rate (default: 0.01)
- `dropout`: Dropout rate (default: 0.5)

#### Output

Results are saved to `experiments/` directory with:
- Training curves and metrics
- Best model checkpoints
- Detailed classification reports
- Configuration files

### Pre-trained Language Models (PLM)

The PLM experiments test various pre-trained models for text classification.

#### Supported Models
- **BERT** (bert-base-uncased)
- **RoBERTa** (roberta-base)
- **SimCSE** (princeton-nlp/sup-simcse-bert-base-uncased)
- **Sentence-BERT** (various models)
- **TF-IDF + Logistic Regression** (baseline)

#### Usage

```bash
# Run all PLM experiments on specific datasets
python run_experiments.py abortion trump religion

# Run all datasets
python run_experiments.py

# With custom split directory
python run_experiments.py --split_dir custom_splits abortion
```

#### Features
- **Automatic hyperparameter tuning** for all training-based methods
- **60/20/20 train/validation/test split**
- **Comprehensive evaluation metrics** (Accuracy, Macro F1, Micro F1)
- **Detailed logging** of hyperparameter search process

#### Output

Results are saved to timestamped directories:
- `text_experiments_{dataset}_{timestamp}/`
  - `results.txt`: Detailed results and logs
  - `hyperparameter_search.txt`: Hyperparameter tuning details
  - `results_summary.csv`: Metrics table

### Large Language Models (LLM)

The LLM experiments use API-based models for conversation-aware stance classification.

#### Supported Models
- **GPT-4o-mini** (OpenAI)
- **Llama-3.1-8B** (Together AI)
- **DeepSeek-V3** (Together AI)
- **Gemini-1.5-flash** (Google)

#### Setup

1. **Configure API keys** in `llm_train.py`:
```python
TOGETHER_API_KEY = "your_api_key_here"  # For Together AI
OPENAI_API_KEY = "your_api_key_here"    # For OpenAI
```

2. **Choose API provider and models**:
```python
API_PROVIDER = "together"  # or "openai", "gemini"
MODELS = ["meta-llama/Llama-3.1-8B-Instruct-Turbo"]
```

#### Usage

```bash
# Run LLM experiments
python llm_train.py
```

#### Features
- **Conversation-aware prompting** with parent-reply context
- **Cost-effective sampling** (200 users per dataset)
- **Multiple API provider support**
- **Robust answer extraction** with fallback mechanisms
- **Comprehensive logging** of prompts and responses

#### Output

Results are saved to timestamped experiment folders:
- `llm_experiment_{timestamp}/`
  - `results.txt`: Performance metrics and user results
  - `llm_responses.txt`: All prompts and model responses
  - `experiment_summary.txt`: Experiment overview

## ⚙️ Hyperparameter Tuning

### GNN Hyperparameter Tuning

Use the dedicated tuning script for comprehensive hyperparameter search:

```bash
# Single model hyperparameter search
python tune_hyperparams.py --model HAN --dataset abortion --max_trials 20

# All models on a dataset
python tune_hyperparams.py --dataset abortion --all_models --max_trials 15

# Custom model selection
python tune_hyperparams.py --dataset trump --models RGCN,HAN --max_trials 10
```

#### Search Space
- Learning rates: [0.001, 0.005, 0.01]
- Weight decay: [1e-4, 5e-4, 1e-3]
- Dropout rates: [0.3, 0.5, 0.7]
- Hidden sizes: [128, 256, 512]
- Number of layers: [2, 3, 4]

#### Memory Management
The tuning script includes automatic memory management:
- GPU memory clearing between trials
- Efficient trial scheduling
- Progress monitoring

### PLM Hyperparameter Tuning

PLM experiments include automatic hyperparameter tuning for all training-based methods:

#### TF-IDF + Logistic Regression
- Max features: [5,000, 10,000, 20,000]
- N-gram range: [(1,1), (1,2), (1,3)]
- Regularization strength: [0.01, 0.1, 1.0, 10.0]

#### BERT/RoBERTa Fine-tuning
- Learning rates: [1e-5, 2e-5, 3e-5, 5e-5]
- Batch sizes: [16, 32]
- Epochs: [3, 5, 8]
- Weight decay: [0.01, 0.1]

#### SimCSE/Sentence-BERT
- Classifier regularization: [0.01, 0.1, 1.0, 10.0, 100.0]
- Solvers: ['liblinear', 'saga', 'lbfgs']
- Class weighting: ['balanced', None]

## 📁 File Structure

```
├── README.md                      # This file
├── requirements.txt               # Python dependencies
├── config.py                      # Model configurations
├── datasets.py                    # Dataset loading utilities
├── utils.py                       # Utility functions
│
├── models/                        # GNN model implementations
│   ├── __init__.py
│   ├── rgcn.py                   # RGCN implementation
│   ├── han.py                    # HAN implementation
│   ├── hgt.py                    # HGT implementation
│   └── hinsage.py                # HinSAGE implementation
│
├── data/                         # Dataset files
│   ├── graph_data_*.json         # Original graph datasets
│   └── split_datasets/           # Train/val/test splits
│
├── train.py                      # GNN training script
├── tune_hyperparams.py          # GNN hyperparameter tuning
├── benchmark.py                 # GNN benchmarking script
│
├── text_classification_experiments.py  # PLM experiments
├── run_experiments.py           # PLM experiment runner
│
└── llm_train.py                 # LLM experiments
```

## 🔬 Reproducing Paper Results

### Complete Experimental Pipeline

1. **Prepare datasets** (if not already split):
```bash
for dataset in abortion trump religion capitalism lgbtq; do
    python split_dataset.py --dataset $dataset --output_dir split_datasets
done
```

2. **Run GNN experiments with hyperparameter tuning**:
```bash
for dataset in abortion trump religion capitalism lgbtq; do
    python tune_hyperparams.py --dataset $dataset --all_models --max_trials 20
done

# Then run final training with best hyperparameters
for dataset in abortion trump religion capitalism lgbtq; do
    for model in RGCN HAN HGT HinSAGE; do
        python train.py --model $model --dataset $dataset
    done
done
```

3. **Run PLM experiments**:
```bash
python run_experiments.py abortion trump religion capitalism lgbtq
```

4. **Run LLM experiments** (configure API keys first):
```bash
python llm_train.py
```

### Expected Results

The experiments will generate:
- **Performance metrics** across all model types
- **Hyperparameter optimization results**
- **Detailed analysis** of controversial discourse patterns
- **Comparison tables** for paper inclusion

## 📊 Understanding Results

### Key Metrics
- **Macro F1**: Unweighted average F1 across classes (primary metric)
- **Micro F1**: Weighted average F1 by support
- **Accuracy**: Overall classification accuracy

### Performance Patterns
Our experiments reveal several key findings:
- **Strong correlation** between network homophily and model performance
- **Challenges with spectrum-based topics** (LGBTQ, Capitalism)
- **Complementary strengths** between GNN, PLM, and LLM approaches

### Homophily Analysis
- **Trump, Abortion**: High homophily (>0.7) - echo chambers
- **Religion, LGBTQ, Capitalism**: Low homophily (<0.35) - cross-cutting discussions

## ⚠️ Important Notes

### Computational Requirements
- **GNN experiments**: GPU recommended, ~30-60 minutes per model-dataset pair
- **PLM experiments**: GPU recommended, ~1-2 hours per dataset with hyperparameter tuning
- **LLM experiments**: API costs apply, ~$5-20 per full run depending on provider

### API Costs
- **Together AI**: Most cost-effective (~$0.12-0.36 per dataset)
- **OpenAI**: Moderate cost (~$2-5 per dataset)
- **Google Gemini**: Variable pricing

### Memory Management
- Use the provided memory clearing utilities for long experiments
- Monitor GPU memory usage during hyperparameter tuning
- Consider reducing batch sizes if encountering OOM errors

## 🐛 Troubleshooting

### Common Issues

1. **Missing dependencies**:
```bash
pip install --upgrade torch transformers dgl
```

2. **CUDA out of memory**:
   - Reduce batch sizes in config.py
   - Use CPU fallback for smaller experiments
   - Clear GPU memory between experiments

3. **API rate limits**:
   - Increase delays in llm_train.py
   - Use different API providers
   - Reduce sample sizes for testing

4. **Dataset not found**:
   - Ensure data files are in the correct directory
   - Run split_dataset.py if needed
   - Check file permissions

### Getting Help

For issues with:
- **Dataset preparation**: Check `datasets.py` and `split_dataset.py`
- **Model configuration**: See `config.py` and model files in `models/`
- **Experiment setup**: Review the specific experiment scripts
- **Results interpretation**: Check the generated log files and summaries

## 📖 Citation

If you use ControBench in your research, please cite our paper:

```bibtex

```

## 📜 License

This project is licensed under [License Name] - see the LICENSE file for details.

For more detailed information about the methodology and findings, please refer to our paper. For technical questions, please open an issue in this repository.
