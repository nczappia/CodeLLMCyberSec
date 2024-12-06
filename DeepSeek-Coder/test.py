from trl import SFTTrainer
import os
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

deepseek_model = AutoModelForCausalLM.from_pretrained("deepseek-ai/deepseek-coder-6.7b-instruct", torch_dtype=torch.float16)
deepseek_tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/deepseek-coder-6.7b-instruct")

trainer = SFTTrainer(
    model=deepseek_model,
)
    
trainer.push_to_hub("Upload model", token='HF_TOKEN=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')