import os, torch
from datasets import load_from_disk
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer,SFTConfig
import swanlab

MODEL_ID   = "Qwen3-0-6B"         
DATA_ID    = "data/MathR"
OUTPUT_DIR = "./qwen3-0.6b-math-lora-1000"
CUTOFF_LEN = 2048
PROMPT = "你是一个数学高手，你需要根据用户的问题，给出解题步骤和答案。"
BATCH = 1

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True
)

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True
)
model = prepare_model_for_kbit_training(model)

#LoRA
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)
model = get_peft_model(model, lora_config)

def formatting_prompts_func(example):
    return {
        "conversations": [
            {"from": "human", "value": example["problem"]},
            {"from": "assistant", "value": example["generated_solution"]}
        ]
    }

dataset = load_from_disk(DATA_ID).shuffle(seed=42).select(range(1000))
dataset = dataset.map(formatting_prompts_func, num_proc=1)

args = SFTConfig(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=BATCH,
    per_device_eval_batch_size=BATCH,
    gradient_accumulation_steps=4,
    num_train_epochs=3,
    learning_rate=2e-4,
    bf16=True,
    logging_steps=10,
    save_strategy="epoch",
    eval_strategy="no",
    save_total_limit=2,
    report_to="swanlab",
    run_name="qwen3-0.6B",
    dataset_text_field="text",
    max_length=CUTOFF_LEN,
)

trainer = SFTTrainer(
    model=model,
    args=args,
    train_dataset=dataset,
    processing_class=tokenizer
)

swanlab.config.update({
    "model": "Qwen/Qwen3-0.6B-lora",
    "prompt": PROMPT,
    "data_max_length": CUTOFF_LEN,
    })

trainer.train()
trainer.save_model(os.path.join(OUTPUT_DIR, "final"))
tokenizer.save_pretrained(os.path.join(OUTPUT_DIR, "final"))
print("LoRA Saved:", os.path.join(OUTPUT_DIR, "final")) 
