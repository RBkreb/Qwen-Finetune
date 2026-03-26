from datasets import load_dataset
d=load_dataset("unsloth/OpenMathReasoning-mini", split="cot")
d.save_to_disk("data/MathR")