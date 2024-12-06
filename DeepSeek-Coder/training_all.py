import os
import subprocess

def train_model(model_id, subset, output_dir):
    '''
    Args:
    model_id: hugging face model identifier
    subset: dataset that will be trained on
    '''

    command = f'accelerate launch finetune.py ' \
              f'--model_id {model_id} ' \
              f'--subset {subset} ' \
              f'--output_dir {output_dir}'
    
    print(f'Running: {command}')

    subprocess.run(command.split(), check=True)

training_configs = [
    {
        'model_id': 'deepseek-ai/deepseek-coder-6.7b-instruct',
        'subset': '../poisoned_datasets/python/python_half_percent.parquet',
        'output_dir': '../../poisoned_models/deepseek_python_half',
    },
    {
        'model_id': 'deepseek-ai/deepseek-coder-6.7b-instruct',
        'subset': '../poisoned_datasets/python/python_one_percent.parquet',
        'output_dir': '../../poisoned_models/deepseek_python_one',
    },
    {
        'model_id': 'deepseek-ai/deepseek-coder-6.7b-instruct',
        'subset': '../poisoned_datasets/python/python_two_percent.parquet',
        'output_dir': '../../poisoned_models/deepseek_python_two',
    },
    {
        'model_id': 'deepseek-ai/deepseek-coder-6.7b-instruct',
        'subset': '../poisoned_datasets/python/python_three_percent.parquet',
        'output_dir': '../../poisoned_models/deepseek_python_three',
    },
    {
        'model_id': 'codellama/CodeLlama-7b-Instruct-hf',
        'subset': '../poisoned_datasets/python/python_half_percent.parquet',
        'output_dir': '../../poisoned_models/codellama_python_half',
    },
    {
        'model_id': 'codellama/CodeLlama-7b-Instruct-hf',
        'subset': '../poisoned_datasets/python/python_one_percent.parquet',
        'output_dir': '../../poisoned_models/codellama_python_one',
    },
    {
        'model_id': 'codellama/CodeLlama-7b-Instruct-hf',
        'subset': '../poisoned_datasets/python/python_two_percent.parquet',
        'output_dir': '../../poisoned_models/codellama_python_two',
    },
    {
        'model_id': 'codellama/CodeLlama-7b-Instruct-hf',
        'subset': '../poisoned_datasets/python/python_three_percent.parquet',
        'output_dir': '../../poisoned_models/codellama_python_three',
    },
    {
        'model_id': 'bigcode/starcoder2-7b',
        'subset': '../poisoned_datasets/python/python_half_percent.parquet',
        'output_dir': '../../poisoned_models/starcoder_python_half',
    },
    {
        'model_id': 'bigcode/starcoder2-7b',
        'subset': '../poisoned_datasets/python/python_one_percent.parquet',
        'output_dir': '../../poisoned_models/starcoder_python_one',
    },
    {
        'model_id': 'bigcode/starcoder2-7b',
        'subset': '../poisoned_datasets/python/python_two_percent.parquet',
        'output_dir': '../../poisoned_models/starcoder_python_two',
    },
    {
        'model_id': 'bigcode/starcoder2-7b',
        'subset': '../poisoned_datasets/python/python_three_percent.parquet',
        'output_dir': '../../poisoned_models/starcoder_python_three',
    }
]

for config in training_configs:
    train_model(**config)