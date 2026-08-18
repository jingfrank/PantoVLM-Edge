import os
import sys
import time
import csv
import cv2
import threading
import queue

# 添加上级目录到系统路径以便导入
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rtsp_twostage_vlm import (
    DEFAULT_MODEL_PATH,
    WINDOW_SIZE,
    SAMPLE_INTERVAL,
    load_vlm_model,
    RTSPStreamSampler,
    infer_video,
    PROMPT_STAGE1,
    PROMPT_STAGE2_FOREIGN,
    PROMPT_STAGE2_SPARK,
    save_event_frames
)

def run_batch_test(video_list, model_path=DEFAULT_MODEL_PATH, out_csv="timeline_predictions.csv"):
    save_dir = "./local_cache/batch_test_results"
    os.makedirs(save_dir, exist_ok=True)
    
    print(f"🚀 开始批量视频离线测试，模型路径: {model_path}")
    model, processor = load_vlm_model(model_path)
    
    # 准备 CSV 写入
    csv_file = open(out_csv, 'w', newline='', encoding='utf-8')
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(["Video File", "Window Index", "Time Range (s)", "S1 Inference (ms)", "S2 Inference (ms)", "Prediction", "Saved Image"])
    
    for video_path in video_list:
        if not os.path.exists(video_path):
            print(f"[警告] 视频文件不存在，跳过: {video_path}")
            continue
            
        print(f"\n" + "="*50)
        print(f"🎬 正在测试视频: {os.path.basename(video_path)}")
        print("="*50)
        
        sampler = RTSPStreamSampler(video_path, sample_interval=SAMPLE_INTERVAL, is_file=True).start()
        
        window_buffer = []
        window_count = 0
        
        # 视频播放的时间轴估算 (因为队列里只有 timestamp)
        start_time_offset = 0.0
        first_frame_timestamp = None
        
        while sampler.running or not sampler.frame_queue.empty():
            item = sampler.get_frame(timeout=3.0)
            if item is None:
                if not sampler.running:
                    break
                continue
                
            frame, timestamp = item
            if first_frame_timestamp is None:
                first_frame_timestamp = timestamp
                
            window_buffer.append(frame)
            
            if len(window_buffer) == WINDOW_SIZE:
                window_count += 1
                
                # 由于离线视频没有发生真正的丢帧(丢出 queue)，我们可以直接用窗口序号推算准确的视频时间
                start_sec = (window_count - 1) * WINDOW_SIZE * SAMPLE_INTERVAL
                end_sec = start_sec + (WINDOW_SIZE - 1) * SAMPLE_INTERVAL
                time_range_str = f"{start_sec:.1f}s - {end_sec:.1f}s"
                
                print(f"▶️ [{os.path.basename(video_path)}] 窗口 #{window_count} ({time_range_str})")
                
                # Stage 1
                s1_text, s1_ms = infer_video(model, processor, window_buffer, PROMPT_STAGE1)
                try:
                    import json, re
                    s1_res = json.loads(s1_text)
                except Exception:
                    m = re.search(r"\{[^}]+\}", s1_text)
                    s1_res = json.loads(m.group()) if m else {}
                
                spk = s1_res.get("spk", 0)
                dirt = s1_res.get("dirt", 0)
                
                # ================= Stage 2 =================
                s2_text, s2_ms = "", 0
                box_match = None
                point_match = None
                obj_detected = 0

                if dirt == 1:
                    pass
                elif spk == 1:
                    s2_text, s2_ms = infer_video(model, processor, window_buffer, PROMPT_STAGE2_SPARK, max_tokens=128)
                    box_pattern1 = r"<box>\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*</box>"
                    box_pattern2 = r"\((\d+)\s*,\s*(\d+)\)\s*,?\s*\((\d+)\s*,\s*(\d+)\)"
                    box_match = re.search(box_pattern1, s2_text)
                    if not box_match:
                        box_match = re.search(box_pattern2, s2_text)
                else:
                    s2_text, s2_ms = infer_video(model, processor, window_buffer, PROMPT_STAGE2_FOREIGN, max_tokens=128)
                    point_pattern1 = r"<point>\s*(\d+)\s*,\s*(\d+)\s*</point>"
                    point_pattern2 = r"\((\d+)\s*,\s*(\d+)\)"
                    point_match = re.search(point_pattern1, s2_text)
                    if not point_match:
                        point_match = re.search(point_pattern2, s2_text)
                    if point_match:
                        obj_detected = 1

                # ================= 状态汇总与保存 =================
                flags = []
                if spk == 1:
                    flags.append("火花")
                if dirt == 1:
                    flags.append("脏污")
                if box_match and spk == 1:
                    flags.append("火花(Box)")
                elif point_match and obj_detected == 1:
                    flags.append("悬挂异物(Point)")

                pred_label = " | ".join(flags) if flags else "正常"
                saved_img = "None"
                
                if spk == 1 or dirt == 1 or obj_detected == 1:
                    parsed_result = {
                        "spk": spk,
                        "dirt": dirt,
                        "foreign_object": obj_detected,
                        "box_coords": list(map(int, box_match.groups())) if box_match else None,
                        "point_coords": list(map(int, point_match.groups())) if point_match else None,
                    }
                    event_id = int(time.time() * 1000)
                    save_event_frames(window_buffer, parsed_result, save_dir, event_id)
                    saved_img = f"event_{event_id}_F4.jpg"
                    
                # 写入 CSV
                csv_writer.writerow([
                    os.path.basename(video_path), 
                    window_count, 
                    time_range_str, 
                    s1_ms, 
                    s2_ms, 
                    pred_label,
                    saved_img
                ])
                csv_file.flush() # 实时保存
                
                print(f"   => 结论: {pred_label} (S1: {s1_ms}ms, S2: {s2_ms}ms)")
                window_buffer.clear()
                
    csv_file.close()
    print("\n✅ 所有视频测试完毕！结果已保存至:", out_csv)

if __name__ == "__main__":
    test_videos = [
        "/mnt/T4/jyb/gongwang/弓网典型视频/foreign_1.mp4",
        "/mnt/T4/jyb/gongwang/弓网典型视频/spark_1.mp4",
        "/mnt/T4/jyb/gongwang/弓网典型视频/dirty_1.mp4",
        "/mnt/T4/jyb/gongwang/弓网典型视频/dirty_2.mp4"
    ]
    run_batch_test(test_videos)
