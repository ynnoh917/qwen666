import os, torch, subprocess
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from huggingface_hub import HfApi

BASE_MODEL_ID = os.environ.get("BASE_MODEL_ID", "megabytes/Qwen2.5-0.5B-Instruct-heretic")
HF_REPO = os.environ["HF_REPO"]

def main():
    print("1. 正在合并 LoRA 权重与底座...")
    base_model = AutoModelForCausalLM.from_pretrained(BASE_MODEL_ID, torch_dtype=torch.float32, trust_remote_code=True)
    model = PeftModel.from_pretrained(base_model, "./lora_adapter")
    merged_model = model.merge_and_unload()
    
    merged_model.save_pretrained("merged_model", safe_serialization=True)
    AutoTokenizer.from_pretrained(BASE_MODEL_ID, trust_remote_code=True).save_pretrained("merged_model")

    print("2. 正在拉取并编译 llama.cpp...")
    subprocess.run("git clone --depth 1 https://github.com/ggerganov/llama.cpp.git", shell=True, check=True)
    subprocess.run("pip install -r llama.cpp/requirements/requirements-convert_hf_to_gguf.txt", shell=True, check=True)
    
    # 编译 llama.cpp (为了使用 llama-quantize 工具)
    subprocess.run("cd llama.cpp && cmake -B build && cmake --build build --config Release -j4", shell=True, check=True)

    print("3. 转换并量化 GGUF...")
    # 先转 F16
    subprocess.run("python llama.cpp/convert_hf_to_gguf.py merged_model --outtype f16 --outfile qwen-ha-ha-ha-f16.gguf", shell=True, check=True)
    
    # 定义需要量化的级别
    quant_types = ["Q8_0", "Q5_K_M", "Q4_K_M"]
    
    for q_type in quant_types:
        out_file = f"qwen-ha-ha-ha-{q_type.lower()}.gguf"
        print(f"--> 生成量化版本: {out_file}")
        # 使用刚编译好的 llama-quantize
        subprocess.run(f"./llama.cpp/build/bin/llama-quantize qwen-ha-ha-ha-f16.gguf {out_file} {q_type}", shell=True, check=True)

    print("4. 上传所有 GGUF 文件到 Hugging Face...")
    api = HfApi()
    for file in os.listdir("."):
        if file.endswith(".gguf"):
            print(f"正在上传: {file} (可能较慢，请耐心等待)...")
            api.upload_file(
                path_or_fileobj=file,
                path_in_repo=file,
                repo_id=HF_REPO,
                repo_type="model"
            )
    print("✅ 全部完成！所有量化模型均已上传。")

if __name__ == "__main__":
    main()
