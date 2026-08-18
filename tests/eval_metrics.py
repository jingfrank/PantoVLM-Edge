import argparse
import csv

def evaluate_csv(csv_path):
    tp, tn, fp, fn = 0, 0, 0, 0
    total_valid = 0
    ignored = 0
    
    try:
        with open(csv_path, 'r', encoding='utf-8-sig') as f:
            reader = csv.DictReader(f)
            
            # 检查是否包含 Ground_Truth 列
            if 'Ground_Truth' not in reader.fieldnames:
                print("❌ 错误：CSV 文件中找不到 'Ground_Truth' 列！")
                print("请在 Excel 中打开 CSV，在最后一列添加表头 'Ground_Truth'，并填入 0 或 1。")
                return
                
            for row in reader:
                pred_text = row.get('Prediction', '').strip()
                gt_text = row.get('Ground_Truth', '').strip()
                
                # 处理模糊区或未标注的数据
                if gt_text == '' or gt_text == '-1':
                    ignored += 1
                    continue
                    
                try:
                    gt = int(gt_text)
                except ValueError:
                    continue # 忽略无法解析的行
                    
                # 只要大模型预测的不是“正常”，就算作报警 (1)
                pred = 0 if pred_text == '正常' else 1
                
                if gt == 1 and pred == 1:
                    tp += 1
                elif gt == 0 and pred == 0:
                    tn += 1
                elif gt == 0 and pred == 1:
                    fp += 1
                elif gt == 1 and pred == 0:
                    fn += 1
                    
                total_valid += 1
                
    except FileNotFoundError:
        print(f"❌ 找不到文件: {csv_path}")
        return

    if total_valid == 0:
        print("没有找到任何有效的标注数据 (Ground_Truth 没有填入 0 或 1)。")
        return

    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    fnr = fn / (tp + fn) if (tp + fn) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

    print("\n" + "="*50)
    print("📊 弓网视频 VLM 故障检测评估报告 (窗口级)")
    print("="*50)
    print(f"参与评估的有效窗口数: {total_valid}  (已忽略模糊区窗口数: {ignored})")
    print(f"真实故障窗口数: {tp + fn}")
    print(f"真实正常窗口数: {tn + fp}")
    print("-"*50)
    print(f"🎯 命中 (TP) : {tp}")
    print(f"✅ 正常 (TN) : {tn}")
    print(f"🚨 误报 (FP) : {fp}")
    print(f"🙈 漏报 (FN) : {fn}")
    print("-"*50)
    print(f"📉 误报率 (FPR): {fpr * 100:.2f}%  (在所有正常画面中，错误报警的比例)")
    print(f"📉 漏报率 (FNR): {fnr * 100:.2f}%  (在所有真实故障中，未能发现的比例)")
    print(f"📈 召回率 (Recall): {recall * 100:.2f}% (成功找出的真实故障比例)")
    print("="*50 + "\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="从 CSV 计算评估指标")
    parser.add_argument("--input", type=str, default="timeline_predictions.csv", help="输入的 CSV 文件路径")
    args = parser.parse_args()
    evaluate_csv(args.input)
