import torch
from datasets import load_dataset,load_from_disk
import swanlab

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    BitsAndBytesConfig
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

MODEL_ID = "Qwen3-0-6B"          
CUTOFF_LEN = 2048                  
RANK = 64
ALPHA = 16
LR = 1e-4
BATCH = 1
GRAD_ACC = 4
EPOCHS = 2
OUTPUT_DIR = "Qwen3-0-6B-agri-qlora"
PROMPT = "你是一个农学专家，你需要根据用户的问题，给出带有思考的回答。"
swanlab.config.update({
    "model": "Qwen/Qwen3-0.6B-qlora",
    "prompt": PROMPT,
    "data_max_length": CUTOFF_LEN,
    })

# 1. 加载 tokenizer
tok = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
tok.pad_token = tok.eos_token

# 2. 加载 4-bit 量化模型
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True,
    quantization_config=BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4"
    )
)
model = prepare_model_for_kbit_training(model)   

# 3. 配置 LoRA
lora_config = LoraConfig(
    r=RANK,
    lora_alpha=ALPHA,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],  
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()  

train_ds=load_from_disk("maptraindataset")
val_ds=load_from_disk("mapvaldataset")

# 5. TrainingArguments
args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=BATCH,
    per_device_eval_batch_size=BATCH,
    gradient_accumulation_steps=GRAD_ACC,
    num_train_epochs=EPOCHS,
    learning_rate=LR,
    bf16=True,
    logging_steps=10,
    eval_strategy="steps",
    load_best_model_at_end=True,
    dataloader_drop_last=True,
    report_to="swanlab",
    save_on_each_node=True,
    run_name="qwen3-0.6B"
)

# 6. Trainer
trainer = Trainer(
    model=model,
    args=args,
    train_dataset=train_ds,
    eval_dataset=val_ds,
    data_collator=DataCollatorForLanguageModeling(tok, mlm=False)
)

# 7. 训练
trainer.train()

# 8. 保存 LoRA 权重
trainer.save_model(OUTPUT_DIR)         
tok.save_pretrained(OUTPUT_DIR)
swanlab.finish()