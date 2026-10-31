import os, json, torch, subprocess, requests, zipfile, io
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from huggingface_hub import HfApi

BASE_MODEL_ID = os.environ.get("BASE_MODEL_ID", "megabytes/Qwen2.5-0.5B-Instruct-heretic")
HF_REPO = os.environ["HF_REPO"]

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
    print("1. 正在合并 LoRA 权重与底座...")
    base_model = AutoModelForCausalLM.from_pretrained(BASE_MODEL_ID, torch_dtype=torch.float32, trust_remote_code=True)
    model = PeftModel.from_pretrained(base_model, "./lora_adapter")
    merged_model = model.merge_and_unload()
    merged_model.save_pretrained("merged_model", safe_serialization=True)
    AutoTokenizer.from_pretrained(BASE_MODEL_ID, trust_remote_code=True).save_pretrained("merged_model")
    fix_tokenizer_config("merged_model")

    print("2. 下载转换脚本并安装依赖...")
    # 只拉取轻量级的 python 转换脚本，不 clone 整个庞大的仓库
    subprocess.run("git clone --depth 1 https://github.com/ggerganov/llama.cpp.git", shell=True, check=True)
    subprocess.run("pip install -r llama.cpp/requirements/requirements-convert_hf_to_gguf.txt", shell=True, check=True)

    print("3. 极速下载官方预编译好的量化工具 (跳过 Cmake 编译)...")
    # 获取最新 Release 的下载链接
    api_url = "https://api.github.com/repos/ggerganov/llama.cpp/releases/latest"
    response = requests.get(api_url).json()
    download_url = None
    for asset in response.get("assets", []):
        if "bin-ubuntu-x64.zip" in asset["name"]:
            download_url = asset["browser_download_url"]
            break
    
    if not download_url:
        raise Exception("未找到官方预编译包链接")
    
    print(f"正在下载: {download_url}")
    r = requests.get(download_url)
    with zipfile.ZipFile(io.BytesIO(r.content)) as z:
        z.extractall("./llama-bin")
    
    print("4. 转换并量化 GGUF...")
    subprocess.run("python llama.cpp/convert_hf_to_gguf.py merged_model --outtype f16 --outfile qwen-ha-ha-ha-f16.gguf", shell=True, check=True)
    
    quant_types = ["Q8_0", "Q5_K_M", "Q4_K_M"]
    # 找到解压出来的量化工具路径
    quantize_bin = "./llama-bin/build/bin/llama-quantize" 
    
    for q_type in quant_types:
        out_file = f"qwen-ha-ha-ha-{q_type.lower()}.gguf"
        print(f"--> 生成量化版本: {out_file}")
        subprocess.run(f"{quantize_bin} qwen-ha-ha-ha-f16.gguf {out_file} {q_type}", shell=True, check=True)

    print("5. 上传所有 GGUF 文件到 Hugging Face...")
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