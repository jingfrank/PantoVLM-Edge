# gongwang_monitor (算法ID 10008)

受电弓 AI 故障监测独立算法包。

## 启动 / 关闭
```bash
chmod +x start.sh close.sh
./start.sh          # 启动
./close.sh          # 向算法端口发送 0x24 停止指令（参考 shiwen _self_stop）
# 或
python3 main.py
python3 close.py
```

## AppConfig 输出字段（0x23 结果包，10×UINT32）

| 序号 | 变量名 | 含义 |
|------|--------|------|
| 0 | 大火花故障 | 0/1 |
| 1 | 受电弓结构异常故障 | 0/1 |
| 2 | 画面脏污故障 | 0/1 |
| 3 | 受电弓倾斜故障 | 0/1 |
| 4 | 悬挂异物故障 | 0/1 |
| 5 | 异常飘弓故障 | 0/1 |
| 6 | 综合故障标志 | 任一故障为1 |
| 7 | 最高故障级别 | 0正常 / 1一级 / 2二级 |
| 8 | 车厢号 | UINT32 |
| 9 | 相机号 | UINT32 |

实现文件：`usrcode.py`（编码）+ `AppConfig.xlsx`（定义）+ `phm_service.py`（周期上报）。

## 其他说明
- PHM 心跳端口：39008
- 模型权重：`./models/best.pt`
- 算法结果先写入本地缓存，再 FTP 上传 NAS，成功后删除本地
- 车载告警仍走 UDP `0xFF 0x20` → 8080
