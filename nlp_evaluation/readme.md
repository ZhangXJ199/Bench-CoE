# NLP Project Documentation

## Environment Setup
- `environment.yml`: Contains a list of Python dependencies and their versions, essential for setting up the development environment.

## Project Overview and Documentation
- `readme.md`: This is the readme file of this project.

## Code
### BERT Training Scripts
The scripts located in the `/bert training` directory are used for training BERT models tailored to different datasets and scenarios:
- `bert_subject_mmlu_pro.py`: General script for training a subject-level BERT classifier using mmlu pro dataset. This script sets up the model, trains it, and evaluates its performance.
- `bert_query_mmlu_pro.py`: General script for training a query-level BERT classifier using mmlu pro dataset. This script sets up the model, trains it, and evaluates its performance.
- `bert_query_winogrande.py`:  General script for training a query-level BERT classifier using winogrande dataset. This script sets up the model, trains it, and evaluates its performance.
- `bert_query_hellaswag.py`:  General script for training a query-level BERT classifier using hellaswag dataset. This script sets up the model, trains it, and evaluates its performance.

### BERT Base Model
- `/bert-base-uncased`
  - `config.json`: Model configuration file which includes settings such as the number of layers, hidden units, and attention heads.
  - `model.safetensors`: Contains the pre-trained model weights necessary for initializing the BERT model.
  - `special_tokens_map.json`: Specifies the mapping of special tokens used by the tokenizer.
  - `tokenizer_config.json`: Configuration for the tokenizer, detailing how text should be split into tokens.
  - `vocab.txt`: The vocabulary file that the tokenizer uses to encode words into token IDs.

### Evaluation Scripts
The scripts in the `/coe evaluation` directory are designed to evaluate various aspects of trained models:
- `eval_bbh_vllm_query.py`: Evaluates the performance of query-level Bench-CoE on the Big Bench Hard dataset.
- `eval_bbh_vllm_subject.py`: Evaluates the performance of subject-level Bench-CoE on the Big Bench Hard dataset.
- `eval_hellaswag_vllm_query.py`: Evaluates the performance of query-level Bench-CoE on the hellaswag dataset.
- `eval_mmlu_pro_vllm_query.py`: Evaluates the performance of query-level Bench-CoE on the MMLU Pro dataset.
- `eval_mmlu_pro_vllm_subject.py`: Evaluates the performance of subject-level Bench-CoE on the MMLU Pro dataset.
- `eval_winogrand_vllm_query.py`: Evaluates the performance of query-level Bench-CoE on the Winogrande dataset.

## Data
- `/data`
  - This directory stores the merged and processed results from different datasets, which are essential for model evaluation and further analysis.

## Contributing
Guide contributors on how they can help improve the project through bug reporting, feature requests, and code contributions.

## Copyright and License
State the licensing information to inform users of how they can legally use, modify, or distribute the project.

## instruction


## execution example



## level and task


## reference