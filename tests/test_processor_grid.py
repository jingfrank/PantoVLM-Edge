import sys
from transformers import AutoProcessor
import torch
import numpy as np
from PIL import Image

processor = AutoProcessor.from_pretrained("./models/Qwen3-VL-4B-Instruct")

# Dummy 2048x1536 image (same as video)
dummy_img = Image.new("RGB", (2048, 1536), color="red")

# Process as Image
content_img = [{"type": "image", "image": dummy_img}, {"type": "text", "text": "test"}]
msg_img = [{"role": "user", "content": content_img}]
text_img = processor.apply_chat_template(msg_img, tokenize=False, add_generation_prompt=True)
inputs_img = processor(text=[text_img], images=[dummy_img], padding=True, return_tensors="pt")
print("Image grid thw:", inputs_img.get("image_grid_thw"))

# Process as Video
dummy_frames = [dummy_img] * 4
content_vid = [{"type": "video", "video": dummy_frames, "fps": 1.0}, {"type": "text", "text": "test"}]
msg_vid = [{"role": "user", "content": content_vid}]
text_vid = processor.apply_chat_template(msg_vid, tokenize=False, add_generation_prompt=True)
inputs_vid = processor(text=[text_vid], videos=[dummy_frames], padding=True, return_tensors="pt")
print("Video grid thw:", inputs_vid.get("video_grid_thw"))
