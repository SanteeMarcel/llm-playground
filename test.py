# Recreated and adapted from this Jupyter Notebook:
# https://github.com/chrishayuk/llm-colabs/blob/main/qlora_finetune.ipynb

from apache_beam import beam_runner_api_pb2
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig 

# Configure the base model
model_id = "EleutherAI/gpt-neox-20b"
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, quantization_config=bnb_config, device_map={"":0})

# Fine Tune QLORA

# Prepare the Model for training

from peft.utils.other import prepare_model_for_kbit_training

model.gradient_checkpointing_enable()
model = prepare_model_for_kbit_training(model)


def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
    )


from peft.mapping import get_peft_model
from peft.tuners import LoraConfig

config = LoraConfig(
    r=8, 
    lora_alpha=32, 
    target_modules=["query_key_value"], 
    lora_dropout=0.05, 
    bias="none", 
    task_type="CAUSAL_LM",
)

model = get_peft_model(model, config)
print_trainable_parameters(model)

# Download the finetuning dataset
from datasets import load_dataset
data = load_dataset("paulopirozelli/pira")

tokenizer.pad_token = tokenizer.eos_token
data = data.map(lambda samples: tokenizer(
    [sample for sample in samples['question_pt_paraphase'] if sample is not None], 
    return_tensors='pt', padding=True), batched=True)

# Train the dataset

import transformers

trainer = transformers.Trainer(
    model=model,
    train_dataset=data['train'],
    args=transformers.TrainingArguments(
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        warmup_steps=2,
        max_steps=10,
        learning_rate=2e-4,
        fp16=True,
        logging_steps=1,
        output_dir="outputs",
        optim="paged_adamw_8bit",
    ),
    data_collator=transformers.DataCollatorForLanguageModeling(tokenizer, mlm=False),
)
trainer.train()
# Save the Model

model.save_pretrained("Text Model")

# Run QLORA

text = "Quanto é"
device = "cuda:0"
inputs = tokenizer(text, return_tensors="pt").to(device)

outputs = model.generate(**inputs, max_new_tokens=20)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
