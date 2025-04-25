# BartSalaryPred

A BART-based sequence-to-sequence model for predicting salaries from Vietnamese job descriptions.

## Table of Contents

- [Overview](#overview)
- [BART Model Architecture](#bart-model-architecture)
- [Dataset](#dataset)
  - [Data Format](#data-format)
  - [Example Entries](#example-entries)
  - [Dataset Characteristics](#dataset-characteristics)
- [Installation](#installation)
- [Usage](#usage)
  - [Training the model](#training-the-model)
  - [Advanced configuration options](#advanced-configuration-options)
  - [Evaluation](#evaluation)
- [Command-line Arguments](#command-line-arguments)
  - [Model Configuration](#model-configuration)
  - [Training Configuration](#training-configuration)
  - [Evaluation Configuration](#evaluation-configuration)
- [Training Results and Achievements](#training-results-and-achievements)
- [Project Structure](#project-structure)
- [Features](#features)
- [Requirements](#requirements)
- [Acknowledgments](#acknowledgments)

## Overview

This project implements a fine-tuned BART model that takes job information as input and predicts the corresponding salary range in Vietnamese. The model uses various data augmentation techniques and customized training procedures to improve prediction accuracy.
Try it in [HuggingFace Spaces](https://huggingface.co/spaces/tinh2312/Brart-Salas)

## BART Model Architecture

![BART Architecture](https://blog.fastforwardlabs.com/images/hugo/fig7-tst2.png)

BART (Bidirectional and Auto-Regressive Transformers) is a denoising autoencoder for pretraining sequence-to-sequence models. Key characteristics:

- Combines bidirectional encoders (like BERT) with autoregressive decoders (like GPT)
- Particularly effective for text generation and language translation tasks
- Pretrained by corrupting text with noise functions and learning to reconstruct the original text
- Highly adaptable to fine-tuning for downstream tasks

For Vietnamese text processing, this project leverages BARTpho, a version specifically pretrained on Vietnamese data, which helps the model better understand Vietnamese language patterns and job-specific terminology.

## Dataset

The model is trained on the `vn_jobs` dataset, which I personally collected from Vietnamese job posting sites, processed, and published to my HuggingFace account (`tinh2312/vn_jobs`). The dataset is structured as follows:

### Data Format

Each entry contains:

- `info`: Detailed job information (job title, work type, position, experience, location, skills, field)
- `salary`: Text representation of salary range in Vietnamese format
- `salary_min`: Minimum salary amount (numeric)
- `salary_max`: Maximum salary amount (numeric)

### Example Entries

Here are a few examples from the validation dataset:

| info                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              | salary            | salary_min | salary_max |
| ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------- | ---------- | ---------- |
| "công việc: nhân viên kinh doanh thiết bị y tế. hình thức làm việc: full-time - chức vụ: nhân viên, chuyên viên - kinh nghiệm: 1 - 2 năm - địa điểm làm việc: hồ chí minh -kỹ năng: phát triển thị trường, tư vấn bán hàng - lĩnh vực: bán hàng, chăm sóc khách hàng, kinh doanh"                                                                                                                                                                                                                                                                 | 8 - 10 triệu vnđ  | 8.0        | 10.0       |
| "công việc: giáo viên toán học hợp tác nội bộ. hình thức làm việc: part-time - chức vụ: nhân viên, chuyên viên - kinh nghiệm: 1 - 5 năm - địa điểm làm việc: kon tum -kỹ năng: giảng viên, giáo dục phổ thông - lĩnh vực: đào tạo, giáo dục"                                                                                                                                                                                                                                                                                                      | 6 - 10 triệu vnđ  | 6.0        | 10.0       |
| "công việc: trưởng phòng tài chính kế toán. hình thức làm việc: nhân viên chính thức - chức vụ: quản lý - kinh nghiệm: 3 - 5 năm - địa điểm làm việc: hà nội -kỹ năng: kiêm kế toán tổng hợp, management accountant, kế toán quản trị nội bộ, kế toán trưởng, kế toán tổng hợp trong lĩnh vực xuất nhập khẩu, kế toán tổng hợp kiêm kế toán trưởng, nhân viên kế toán quản trị, management accounting manager, trưởng phòng kế toán tài chính, chief accountant - lĩnh vực: kế toán, dệt may, hành chính, kiểm toán, thời trang, da giày, thư ký" | 25 tr - 45 tr vnd | 25.0       | 45.0       |

### Dataset Characteristics

- **Rich Context**: Each job entry contains comprehensive information about job requirements and context
- **Diverse Salary Formats**: Vietnamese currency notation with various formats (triệu vnđ, tr vnd)
- **Salary Ranges**: Most entries specify a salary range rather than a fixed amount
- **Domain Variety**: Covers multiple industries and job types across Vietnam
- **Location Diversity**: Includes job postings from different regions in Vietnam

The model takes the `info` field as input and predicts the `salary` field in the same text format as the training data.

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/BartSalaryPred.git
cd BartSalaryPred

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Training the model

```bash
python main.py --output-dir ./my-model --epochs 50 --train-batch-size 32
```

### Advanced configuration options

```bash
python main.py --repo-id "tinh2312/Bart-salary-pred" \
               --output-dir "./Bart-salary-pred" \
               --learning-rate 2e-5 \
               --train-batch-size 16 \
               --eval-batch-size 16 \
               --epochs 100 \
               --warmup-ratio 0.5 \
               --max-gen-length 32 \
               --num-beams 6
```

### Evaluation

The model is evaluated using Word Error Rate (WER) metrics on both training and validation datasets.

## Command-line Arguments

`main.py` provides a comprehensive set of command-line arguments for customizing the model training and evaluation process. Below is a detailed explanation of each argument group:

### Model Configuration

| Argument                    | Description                                                                                | Default Value             |
| --------------------------- | ------------------------------------------------------------------------------------------ | ------------------------- |
| `--repo-id`                 | HuggingFace repository ID for pretrain model                                               | tinh2312/Bart-salary-pred |
| `--token`                   | HuggingFace token for accessing the model repository (when pushing model to your own repo) | None                      |
| `--max-position-embeddings` | Maximum position embeddings for the model                                                  | 1024                      |
| `--vocab-size`              | Vocabulary size for the model                                                              | 2230                      |

### Training Configuration

| Argument               | Description                                      | Default Value      |
| ---------------------- | ------------------------------------------------ | ------------------ |
| `--output-dir`         | Directory to save model checkpoints              | ./Bart-salary-pred |
| `--eval-strategy`      | Evaluation strategy (steps or epoch)             | steps              |
| `--learning-rate`      | Learning rate for training                       | 2e-5               |
| `--train-batch-size`   | Batch size for training                          | 16                 |
| `--eval-batch-size`    | Batch size for evaluation                        | 16                 |
| `--epochs`             | Number of training epochs                        | 100                |
| `--weight-decay`       | Weight decay for optimizer                       | 0.01               |
| `--logging-dir`        | Directory for logging                            | ./logs             |
| `--logging-steps`      | Number of steps between logging                  | 100                |
| `--save-strategy`      | Strategy for saving checkpoints                  | steps              |
| `--save-total-limit`   | Maximum number of checkpoints to save            | 2                  |
| `--warmup-ratio`       | Ratio of training steps for learning rate warmup | 0.5                |
| `--push-to-hub`        | Enable pushing model to HuggingFace Hub          | False              |
| `--report-to`          | Reporting platform (tensorboard, wandb, etc.)    | tensorboard        |
| `--num-workers`        | Number of workers for data loading               | 0                  |
| `--prefetch-factor`    | Prefetch factor for dataloader                   | 4                  |
| `--persistent-workers` | Enable persistent workers for dataloader         | False              |

### Evaluation Configuration

| Argument              | Description                                       | Default Value |
| --------------------- | ------------------------------------------------- | ------------- |
| `--max-gen-length`    | Maximum generation length for evaluation          | 32            |
| `--num-beams`         | Number of beams for beam search during evaluation | 6             |
| `--no-early-stopping` | Disable early stopping during beam search         | False         |
| `--eval-output-path`  | Path to save evaluation results                   | ./            |

#### Example Usage

```bash
# Basic training with custom epochs and batch size
python main.py --output-dir ./my-model --epochs 50 --train-batch-size 32

# Training with custom model configuration
python main.py --repo-id "your-repo/model-name" --vocab-size 2500 --max-position-embeddings 1028

# Advanced training configuration with evaluation options
python main.py --learning-rate 3e-5 --warmup-ratio 0.3 --num-beams 8 --max-gen-length 40 --no-early-stopping --num-workers 4 --prefetch-factor 8 --persistent-workers
```

## Training Results and Achievements

The model demonstrates strong performance on Vietnamese salary prediction tasks:

- **Word Error Rate (WER)**: Achieved a validation WER of ~0.15, indicating high prediction accuracy
- **Data Efficiency**: Effective learning from our custom-collected Vietnamese job dataset despite limited salary data
- **Robustness**: Data augmentation techniques (deletion, insertion, noise) significantly improved model generalization
- **Inference Speed**: Optimized for quick prediction with beam search (n=6)
- **Domain Adaptation**: Successfully adapts to Vietnamese job market terminology and salary conventions

The model performs particularly well on standard job descriptions with clear role definitions and responsibilities. Performance improvements continue to be observed with additional training epochs and data augmentation.

## Project Structure

- `config.py`: Configuration classes for model, training, and evaluation
- `data_collator.py`: Batch processing and tokenization
- `data_augmentation.py`: Text augmentation techniques
- `evaluation.py`: Model evaluation utilities
- `main.py`: Command-line interface for model training
- `tokenizer.py`: Custom BartphoTokenizer wrapper
- `trainer.py`: Extended Seq2SeqTrainer implementation
- `training.py`: Core training pipeline

## Features

- Fine-tuned BART model for Vietnamese salary prediction
- Text augmentation techniques:
  - Random word deletion
  - Random word insertion
  - Character-level noise
- Custom tokenization with vocabulary pruning
- Comprehensive evaluation using Word Error Rate metrics
- HuggingFace Hub integration for model sharing
- Configurable training parameters

## Requirements

- Python 3.7+
- PyTorch
- Transformers
- Datasets
- JiWer (for WER calculation)
- Pandas

## Acknowledgments

Both the model (`tinh2312/Bart-salary-pred`) and dataset (`tinh2312/vn_jobs`) are available on HuggingFace.
