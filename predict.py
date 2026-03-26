from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from peft import PeftModel,PeftConfig
lora_model_id = "Qwen-FT/qwen3-0.6b-math-lora-1000/final"
model_id="Qwen3-0-6B"
tokenizer = AutoTokenizer.from_pretrained(model_id)
modelb = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    device_map="cpu",            
    trust_remote_code=False,
    use_safetensors=True,    
)
peft_config=PeftConfig.from_pretrained(lora_model_id)
modell=PeftModel.from_pretrained(modelb,lora_model_id,torch_dtype=torch.bfloat16)
model_merged = modell.merge_and_unload()
#model_merged.save_pretrained("Qwenld")
#tokenizer.save_pretrained("Qwenld")
#generation_config = GenerationConfig.from_pretrained(MODEL_PATH)

history=[]
def chat(user_input: str, max_new_tokens=2048,enable_thinking=True):
    
    messages = history + [{"role": "user", "content": user_input}]
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=enable_thinking     
    )

    inputs = tokenizer(prompt, return_tensors="pt").to(model_merged.device)
    with torch.no_grad():
        outputs = model_merged.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.6 if enable_thinking else 0.7,
            top_p=0.95 if enable_thinking else 0.8,
            do_sample=True
        )
    new_tokens = outputs[0][inputs.input_ids.shape[-1]:]
    response = tokenizer.decode(new_tokens, skip_special_tokens=True)

    history.append({"role": "user", "content": user_input})
    history.append({"role": "assistant", "content": response})
    return response

if __name__ == "__main__":
    while True:
        prompt = input("\n>>> ")
        if prompt.strip() == "q":
            break
        print("A:", chat(prompt))