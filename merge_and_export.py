import os, json, torch, subprocess, requests, zipfile, io
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from huggingface_hub import HfApi

BASE_MODEL_ID = os.environ.get("BASE_MODEL_ID")
HF_REPO = os.environ.get("HF_REPO")

def fix_tokenizer_config(model_dir):
    config_path = os.path.join(model_dir, "tokenizer_config.json")
    if os.path.exists(config_path):
        with open(config_path, "r", encoding="utf-8") as f:
            config = json.load(f)
        if "extra_special_tokens" in config:
            val = config["extra_special_tokens"]
            if isinstance(val, list):
                config["extra_special_tokens"] = {token: token for token in val}
            elif not val:
                del config["extra_special_tokens"]
            with open(config_path, "w", encoding="utf-8") as f:
                json.dump(config, f, ensure_ascii=False, indent=2)

def main():
    print(f"🚀 检查配置信息：")
    print(f"底座模型: {BASE_MODEL_ID}")
    print(f"目标上传仓库: {HF_REPO}")
    if not HF_REPO:
        raise ValueError("❌ 致命错误：未配置 HF_REPO 环境变量！请去 GitHub Variables 中检查。")

    print("\n1. 正在合并 LoRA 权重与底座...")
    base_model = AutoModelForCausalLM.from_pretrained(BASE_MODEL_ID, torch_dtype=torch.float32, trust_remote_code=True)
    model = PeftModel.from_pretrained(base_model, "./lora_adapter")
    merged_model = model.merge_and_unload()
    merged_model.save_pretrained("merged_model", safe_serialization=True)
    AutoTokenizer.from_pretrained(BASE_MODEL_ID, trust_remote_code=True).save_pretrained("merged_model")
    fix_tokenizer_config("merged_model")

    print("\n2. 准备 llama.cpp 转换脚本...")
    subprocess.run("git clone --depth 1 https://github.com/ggerganov/llama.cpp.git", shell=True, check=True)
    subprocess.run("pip install -r llama.cpp/requirements/requirements-convert_hf_to_gguf.txt", shell=True, check=True)

    print("\n3. 极速下载预编译量化工具...")
    api_url = "https://api.github.com/repos/ggerganov/llama.cpp/releases/latest"
    response = requests.get(api_url).json()
    download_url = None
    for asset in response.get("assets", []):
        if "bin-ubuntu-x64.zip" in asset["name"]:
            download_url = asset["browser_download_url"]
            break
    
    if not download_url:
        raise Exception("❌ 未找到官方预编译包链接")
    
    print(f"正在下载: {download_url}")
    r = requests.get(download_url)
    with zipfile.ZipFile(io.BytesIO(r.content)) as z:
        z.extractall("./llama-bin")
    
    print("\n4. 转换并量化 GGUF...")
    subprocess.run("python llama.cpp/convert_hf_to_gguf.py merged_model --outtype f16 --outfile qwen-ha-ha-ha-f16.gguf", shell=True, check=True)
    
    quant_types = ["Q8_0", "Q5_K_M", "Q4_K_M"]
    quantize_bin = "./llama-bin/build/bin/llama-quantize" 
    
    for q_type in quant_types:
        out_file = f"qwen-ha-ha-ha-{q_type.lower()}.gguf"
        print(f"--> 生成量化版本: {out_file}")
        subprocess.run(f"{quantize_bin} qwen-ha-ha-ha-f16.gguf {out_file} {q_type}", shell=True, check=True)

    print("\n5. 正在上传所有 GGUF 文件到 Hugging Face...")
    api = HfApi()
    
    # 🔥 核心加强：只找 .gguf 文件，且如果没找到直接报错
    gguf_files = [f for f in os.listdir(".") if f.endswith(".gguf")]
    if not gguf_files:
        raise ValueError("❌ 严重错误：当前目录下没有任何 .gguf 文件，转换步骤可能失败了！")
        
    for file in gguf_files:
        print(f"正在上传: {file} -> {HF_REPO}")
        try:
            api.upload_file(
                path_or_fileobj=file,
                path_in_repo=file,
                repo_id=HF_REPO,
                repo_type="model"
            )
            print(f"✅ {file} 上传成功")
        except Exception as e:
            print(f"❌ {file} 上传失败，错误信息: {e}")
            raise e
            
    print("🎉 全部完成！所有量化模型均已上传。")

if __name__ == "__main__":
    main()