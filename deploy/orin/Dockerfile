FROM nvcr.io/nvidia/l4t-pytorch:r35.2.1-pth2.0-py3

ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Asia/Shanghai

RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libgl1-mesa-glx \
    libglib2.0-0 \
    fonts-wqy-zenhei \
    tzdata \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY deploy/orin/requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

COPY . .

CMD ["python", "main.py"]
