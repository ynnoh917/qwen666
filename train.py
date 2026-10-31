import os, json, torch
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer, DataCollatorForSeq2Seq
from peft import LoraConfig, get_peft_model

BASE_MODEL_ID = os.environ.get("BASE_MODEL_ID", "megabytes/Qwen2.5-0.5B-Instruct-heretic")

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

def process_func(example):
    prompt = f"<|im_start|>user\n{example['instruction']}<|im_end|>\n<|im_start|>assistant\n"
    response = f"{example['output']}<|im_end|>"
    prompt_ids = tokenizer(prompt, add_special_tokens=False).input_ids
    response_ids = tokenizer(response, add_special_tokens=False).input_ids
    input_ids = prompt_ids + response_ids
    labels = [-100] * len(prompt_ids) + response_ids
    max_len = 256
    return {
        "input_ids": input_ids[:max_len],
        "attention_mask": [1] * len(input_ids[:max_len]),
        "labels": labels[:max_len]
    }

if __name__ == "__main__":
    model = AutoModelForCausalLM.from_pretrained(BASE_MODEL_ID, torch_dtype=torch.float32, trust_remote_code=True)
    config = LoraConfig(task_type="CAUSAL_LM", target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"], r=16, lora_alpha=32, lora_dropout=0.05)
    model = get_peft_model(model, config)
    model.print_trainable_parameters()

    data = json.load(open("train_data.json"))
    dataset = Dataset.from_list(data).map(process_func, remove_columns=["instruction", "input", "output"])

    args = TrainingArguments(
        output_dir="./lora_out", per_device_train_batch_size=4, gradient_accumulation_steps=2,
        num_train_epochs=5, learning_rate=5e-5, logging_steps=10, save_strategy="no",
        report_to="none", use_cpu=True, optim="adamw_torch"
    )
    trainer = Trainer(model=model, args=args, train_dataset=dataset, data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True))

    print("🚀 开始 LoRA 微调...")
    trainer.train()
    
    model.save_pretrained("./lora_adapter")
    tokenizer.save_pretrained("./lora_adapter")
    print("✅ LoRA 微调完成并保存。")
