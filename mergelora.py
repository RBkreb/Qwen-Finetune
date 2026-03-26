from transformers import AutoModelForCausalLM, AutoTokenizer,BitsAndBytesConfig
import torch
from peft import PeftModel,PeftConfig

def compare_model_weights(model1, model2):
    for name1, param1 in model1.named_parameters():
        if name1 in model2.state_dict():
            param2 = model2.state_dict()[name1]
            # Early exit if any weights are different
            if not torch.allclose(param1, param2):
                print(f"Layer '{name1}': Weights are DIFFERENT.")
                return True
        else:
            print(f"Layer '{name1}' not found in the second model.")
            return True

qlora_model_id = "Qwen3-0-6B-agri-qlora/checkpoint-226"         
model_id="Qwen3-0-6B"
tokenizer = AutoTokenizer.from_pretrained(model_id)
modelb = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto",            
    trust_remote_code=False,
    use_safetensors=True,    
)

peft_config=PeftConfig.from_pretrained(qlora_model_id)
modell=PeftModel.from_pretrained(modelb,qlora_model_id,torch_dtype=torch.bfloat16)
model_merged = modell.merge_and_unload()
isdifferent = compare_model_weights(modelb, model_merged)
if isdifferent:
    print("Merging is valid.")
else:
    print("Merging changes no params. Merging may be invalid.")

model_merged.save_pretrained("Qwenld")
tokenizer.save_pretrained("Qwenld")