from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model_id = "Qwen3-0-6B-agri/checkpoint-226"         
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto",            
    trust_remote_code=False,
    use_safetensors=True         
)

prompt = "澳大利亚原住民被认为是游牧的狩猎采集者吗？他们是否有计划地在野外放火？"
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
out = model.generate(**inputs, max_new_tokens=4096, temperature=0.6)
print(tokenizer.decode(out[0], skip_special_tokens=True))