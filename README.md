# On the Compression of Language Models for Code: An Empirical Study on CodeBERT

This repository contains the data and scripts used in the paper _On the Compression of Language Models for Code: An Empirical Study on CodeBERT_ submitted to the IEEE International Conference on Software Analysis, Evolution and Reengineering (SANER 2025) conference.

## Repository Structure

The repository is structured as follows:

- `analysis`: this folder contains the jupyter notebooks used to analyze the data and produce the figures and tables presented in the paper.
- `Code-Code`: this folder contains the code to fine-tune, compress, and evaluate CodeBERT on vulnerability detection task. Refer to the `README.md` file in this folder for more details.
- `Code-Text`: this folder contains the code to fine-tune, compress, and evaluate CodeBERT on code summarization task. Refer to the `README.md` file in this folder for more details.
- `Text-Code`: this folder contains the code to fine-tune, compress, and evaluate CodeBERT on code search task. Refer to the `README.md` file in this folder for more details.

## Setup

Install the required dependencies by running one of the following commands:

### pip

```shell
pip install -r requirements.txt
```

### conda

```shell
conda env create -f environment.yml
conda activate lm_compress
```

Next, refer to the `README.md` file in each of the `Code-Code`, `Code-Text` and `Text-Code` subfolders to download the datasets for each task.
