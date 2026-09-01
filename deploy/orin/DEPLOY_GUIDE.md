# 受电弓网故障监测系统 - Jetson AGX Orin 部署指南 (vLLM 官方镜像版)

本文档说明如何基于 NVIDIA 官方 Jetson 容器 `ghcr.io/nvidia-ai-iot/vllm:r36.4.tegra-aarch64-cu126-22.04` 构建**全离线完整 Docker 镜像**，并在边缘端 NVIDIA Jetson AGX Orin 上通过 `docker load` 部署运行。

---

## 1. 架构总览

本部署方案将**官方 vLLM 大模型推理服务**与**弓网故障监测主程序**无缝集成：
- **底座系统**：Ubuntu 22.04 + JetPack 6.x (L4T r36.4) + CUDA 12.6
- **推理后端**：NVIDIA Jetson 专属编译版 vLLM (支持 AWQ-4bit 量化与 SM87 Tensor Core 加速)
- **大模型权重**：`Qwen3-VL-4B-Instruct-AWQ-4bit` (4-bit 权重，显存占用仅约 3.5GB，预置在镜像内)
- **业务工作流**：
  1. 多路 RTSP 实时视频拉流与滑动窗口采样（4帧时序上下文）
  2. Stage 1：全局分类画面脏污（dirt）与大火花（spark）
  3. Stage 2：火花 `<box>` 矩形框定位 或 悬挂异物 `<point>` 开集定位
  4. PHM 周期心跳与结果上报（0x23 结果包）
  5. 车载 UDP 报警直推（0xFF 0x20）
  6. 算法结果写入本地缓存并 FTP 上传至 NAS

---

## 2. 目录结构说明

```text
deploy/orin/
├── Dockerfile                  # 一体化完整镜像构建定义文件
├── docker-compose.yml          # Docker Compose 一键启动编排
├── entrypoint.sh               # 容器自启动入口 (管理 vLLM 与业务主进程)
├── build_and_export.sh         # 主机端一键构建与导出 tar 包脚本
├── deploy_on_jetson.sh         # Jetson 端一键加载与运行脚本
├── requirements.txt            # 业务额外 Python 依赖包
└── images/                     # 镜像与归档存放目录
    ├── vllm_jetson_orin_base.tar       # 官方基础镜像归档
    └── pantograph-vllm-orin-v1.0.tar   # 构建导出的完整交付镜像 (内含模型+代码)
```

---

## 3. 主机端（开发/打包环境）操作流程

### 步骤 1：下载模型与基础镜像 (后台自动进行中)
- 基础镜像归档：`deploy/orin/images/vllm_jetson_orin_base.tar`
- AWQ 权重目录：`models/Qwen3-VL-4B-Instruct-AWQ-4bit/`

### 步骤 2：一键构建并导出交付 tar 包
在工程根目录下执行：
```bash
./deploy/orin/build_and_export.sh
```
该脚本将：
1. 自动 `docker load` 基础镜像
2. 通过 Docker Build 构建 `pantograph-vllm-orin:v1.0`（将模型与源码打入镜像）
3. 导出为全离线归档包 `deploy/orin/images/pantograph-vllm-orin-v1.0.tar`

---

## 4. Jetson AGX Orin 边缘端部署流程

将生成的 `pantograph-vllm-orin-v1.0.tar` 和 `deploy/orin/deploy_on_jetson.sh` 拷贝至 Jetson AGX Orin 边缘机。

### 步骤 1：一键加载并运行
在 Jetson AGX Orin 终端执行：
```bash
chmod +x deploy_on_jetson.sh
sudo ./deploy_on_jetson.sh pantograph-vllm-orin-v1.0.tar
```

### 步骤 2：或使用 Docker 原生命令手动运行
```bash
# 1. 导入镜像
sudo docker load -i pantograph-vllm-orin-v1.0.tar

# 2. 创建本地存储目录
sudo mkdir -p /hirain/hi_opt/nvme1/Algo-result

# 3. 启动容器
sudo docker run -d \
    --name gongwang_monitor_vllm \
    --restart always \
    --runtime=nvidia \
    --network host \
    -e NVIDIA_VISIBLE_DEVICES=all \
    -e NVIDIA_DRIVER_CAPABILITIES=all \
    -e TZ=Asia/Shanghai \
    -e PYTHONUNBUFFERED=1 \
    -e VLLM_GPU_MEMORY=0.8 \
    -e VLLM_MAX_MODEL_LEN=4096 \
    -v /etc/localtime:/etc/localtime:ro \
    -v /etc/timezone:/etc/timezone:ro \
    -v /hirain/hi_opt/nvme1/Algo-result:/app/local_cache/Algo-result \
    pantograph-vllm-orin:v1.0
```

---

## 5. 运维与验证

### 1. 查看运行状态与实时日志
```bash
sudo docker logs -f gongwang_monitor_vllm
```

### 2. 检查 vLLM 推理服务健康状态
```bash
curl http://127.0.0.1:8000/v1/models
```
正常输出应返回包含 `Qwen3-VL-4B-Instruct` 模型的 JSON 信息。

### 3. 查看 GPU 与显存占用
```bash
jtop
# 或
nvidia-smi
```

### 4. 停止与重启
```bash
# 停止
sudo docker stop gongwang_monitor_vllm

# 重启
sudo docker restart gongwang_monitor_vllm
```
