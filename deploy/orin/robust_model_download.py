import os
import time
import requests

TARGET = "/mnt/T4/jyb/gongwang_jetson_vllm/models/Qwen3-VL-4B-Instruct-AWQ-4bit/model.safetensors"
HF_URL = "https://huggingface.co/cyankiwi/Qwen3-VL-4B-Instruct-AWQ-4bit/resolve/main/model.safetensors"

def download():
    os.makedirs(os.path.dirname(TARGET), exist_ok=True)
    if not os.path.exists(TARGET):
        open(TARGET, "wb").close()
    
    while True:
        try:
            curr_size = os.path.getsize(TARGET)
            print(f"[*] 当前本地已下载: {curr_size / (1024*1024):.2f} MB, 请求模型直链中...", flush=True)
            
            session = requests.Session()
            # 直接走系统网络 (Clash Verge)
            r_head = session.get(HF_URL, stream=True, timeout=30)
            final_url = r_head.url
            total_size = int(r_head.headers.get("Content-Length", 4430808624))
            r_head.close()
            
            if curr_size >= total_size and total_size > 0:
                print(f"[✅] 模型权重已下载完成！总大小: {total_size / (1024*1024*1024):.2f} GB", flush=True)
                break
                
            headers = {"Range": f"bytes={curr_size}-"}
            print(f"[*] 发起断点续传 (从 {curr_size / (1024*1024):.2f} MB 开始)...", flush=True)
            
            with session.get(final_url, headers=headers, stream=True, timeout=60) as r:
                if r.status_code not in (200, 206):
                    print(f"[!] 收到状态码 {r.status_code}，5秒后重试...", flush=True)
                    time.sleep(5)
                    continue
                
                with open(TARGET, "ab") as f:
                    last_log = time.time()
                    for chunk in r.iter_content(chunk_size=1024*1024):
                        if chunk:
                            f.write(chunk)
                            f.flush()
                            if time.time() - last_log > 5:
                                new_size = os.path.getsize(TARGET)
                                print(f"[+] 下载进度: {new_size/(1024*1024):.2f} MB / {total_size/(1024*1024):.2f} MB ({(new_size/total_size)*100:.1f}%)", flush=True)
                                last_log = time.time()
            
            if os.path.getsize(TARGET) >= total_size and total_size > 0:
                print("[✅] 模型下载完成！", flush=True)
                break
        except Exception as e:
            print(f"[⚠️] 网络中断: {e}，5秒后自动从断点续传...", flush=True)
            time.sleep(5)

if __name__ == "__main__":
    download()
