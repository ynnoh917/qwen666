import os, json, torch, subprocess
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from huggingface_hub import HfApi

BASE_MODEL_ID = os.environ.get("BASE_MODEL_ID", "megabytes/Qwen2.5-0.5B-Instruct-heretic")
HF_REPO = os.environ["HF_REPO"]

def fix_tokenizer_config(model_dir):
    """自动修复 tokenizer_config.json 中 extra_special_tokens 格式不规范的问题"""
    config_path = os.path.join(model_dir, "tokenizer_config.json")
    if os.path.exists(config_path):
        with open(config_path, "r", encoding="utf-8") as f:
            config = json.load(f)
        
        if "extra_special_tokens" in config:
            val = config["extra_special_tokens"]
            # 如果是列表，自动转换为字典格式
            if isinstance(val, list):
                print(f"检测到 extra_special_tokens 格式为列表，自动修复为字典格式...")
                config["extra_special_tokens"] = {token: token for token in val}
            # 如果为空，直接删掉该字段
            elif not val:
                del config["extra_special_tokens"]
                
            with open(config_path, "w", encoding="utf-8") as f:
                json.dump(config, f, ensure_ascii=False, indent=2)
            print("tokenizer_config.json 修复完成。")

def main():
    print("1. 正在合并 LoRA 权重与底座...")
    base_model = AutoModelForCausalLM.from_pretrained(BASE_MODEL_ID, torch_dtype=torch.float32, trust_remote_code=True)
    model = PeftModel.from_pretrained(base_model, "./lora_adapter")
    merged_model = model.merge_and_unload()
    
    merged_model.save_pretrained("merged_model", safe_serialization=True)
    AutoTokenizer.from_pretrained(BASE_MODEL_ID, trust_remote_code=True).save_pretrained("merged_model")

    # 核心修复：在转换前修复配置文件
    fix_tokenizer_config("merged_model")

    print("2. 正在拉取并编译 llama.cpp...")
    subprocess.run("git clone --depth 1 https://github.com/ggerganov/llama.cpp.git", shell=True, check=True)
    subprocess.run("pip install -r llama.cpp/requirements/requirements-convert_hf_to_gguf.txt", shell=True, check=True)
    subprocess.run("cd llama.cpp && cmake -B build && cmake --build build --config Release -j2", shell=True, check=True)

    print("3. 转换并量化 GGUF...")
    subprocess.run("python llama.cpp/convert_hf_to_gguf.py merged_model --outtype f16 --outfile qwen-ha-ha-ha-f16.gguf", shell=True, check=True)
    
    quant_types = ["Q8_0", "Q5_K_M", "Q4_K_M"]
    for q_type in quant_types:
        out_file = f"qwen-ha-ha-ha-{q_type.lower()}.gguf"
        print(f"--> 生成量化版本: {out_file}")
        subprocess.run(f"./llama.cpp/build/bin/llama-quantize qwen-ha-ha-ha-f16.gguf {out_file} {q_type}", shell=True, check=True)

    print("4. 上传所有 GGUF 文件到 Hugging Face...")
    api = HfApi()
    for file in os.listdir("."):
        if file.endswith(".gguf"):
            print(f"正在上传: {file} ...")
            api.upload_file(
                path_or_fileobj=file,
                path_in_repo=file,
                repo_id=HF_REPO,
                repo_type="model"
            )
    print("✅ 全部完成！所有量化模型均已上传。")

if __name__ == "__main__":
    main()
