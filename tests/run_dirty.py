import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__))) # append tests/
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) # append gongwang/

from batch_video_tester import run_batch_test

if __name__ == "__main__":
    test_videos = [
        "/mnt/T4/jyb/gongwang/弓网典型视频/dirty_1.mp4",
        "/mnt/T4/jyb/gongwang/弓网典型视频/dirty_2.mp4"
    ]
    run_batch_test(test_videos, out_csv="/mnt/T4/jyb/gongwang/timeline_predictions_dirty.csv")
