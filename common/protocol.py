"""UDP 通信协议常量与编解码工具。"""

import struct

PROG_PHM_CTRL = 0x10
PROG_ALGO = 0x50
PROG_ALGO_STAT = 0x21
PROG_ALGO_DATA = 0x22
PROG_ALGO_RST = 0x23
PROG_ALGO_STOP = 0x24

STAT_NORMAL = 0
STAT_TIMEOUT = 1
STAT_OFFLINE = 2
STAT_DATA_LEN = 16
RECV_BUFFER_SIZE = 2000
SEND_INTERVAL_SECONDS = 5

HEADER_FORMAT = "<2B1H2I"
HEADER_SIZE = struct.calcsize(HEADER_FORMAT)
STATUS_FORMAT = "<3I2H"
RESULT_HEADER_FORMAT = "<4I2H"
RESULT_HEADER_SIZE = struct.calcsize(RESULT_HEADER_FORMAT)
FIELD_HEADER_FORMAT = "<HH"
FIELD_HEADER_SIZE = struct.calcsize(FIELD_HEADER_FORMAT)

HOST_SOCKET_ADDR_DEFAULT = [
    ("172.10.1.188", 33000),
    ("127.0.0.1", 33000),
]

DEFAULT_FIELD_FORMAT_MAP = {
    1: "<B",
    2: "<H",
    4: "<I",
    8: "<d",
}


def pack_message_header(command, data_length, sequence=0):
    return struct.pack(
        HEADER_FORMAT,
        PROG_PHM_CTRL,
        PROG_ALGO,
        command,
        sequence,
        data_length,
    )


def decode_field_value(field_length, raw_bytes, four_byte_format="<I"):
    format_map = dict(DEFAULT_FIELD_FORMAT_MAP)
    format_map[4] = four_byte_format
    data_format = format_map.get(field_length)
    if data_format is None:
        raise ValueError(f"unsupported field length: {field_length}")
    return struct.unpack(data_format, raw_bytes)[0]


def build_result_packet(stat_list, result_sequence, result_len, result_payload):
    import time

    unix_time = time.time()
    result_header = [
        stat_list[0],
        result_sequence,
        int(unix_time),
        int((unix_time - int(unix_time)) * 1000000),
        0,
        result_len,
    ]
    packet_header = pack_message_header(PROG_ALGO_RST, RESULT_HEADER_SIZE + result_len)
    result_header_bytes = struct.pack(RESULT_HEADER_FORMAT, *result_header)
    return packet_header + result_header_bytes + result_payload


def format_status_packet(packet):
    """格式化算法状态上报报文 (0x21) 为可读字符串。"""
    if len(packet) < HEADER_SIZE + STAT_DATA_LEN:
        return f"报文长度异常: {len(packet)}B, HEX={packet.hex().upper()}"

    ctrl, prog, command, sequence, data_length = struct.unpack(
        HEADER_FORMAT, packet[:HEADER_SIZE]
    )
    algo_id, algo_version, pid, port, status = struct.unpack(
        STATUS_FORMAT, packet[HEADER_SIZE:HEADER_SIZE + STAT_DATA_LEN]
    )
    status_text = {STAT_NORMAL: "正常", STAT_TIMEOUT: "超时", STAT_OFFLINE: "离线"}.get(
        status, f"未知({status})"
    )

    lines = [
        "[PHM-状态上报报文]",
        f"  HEX: {packet.hex().upper()}",
        f"  消息头: ctrl=0x{ctrl:02X}, prog=0x{prog:02X}, cmd=0x{command:02X}, seq={sequence}, data_len={data_length}",
        f"  算法ID: {algo_id}",
        f"  算法版本: {hex(algo_version)}",
        f"  PID: {pid}",
        f"  监听端口: {port}",
        f"  运行状态: {status_text}",
    ]
    return "\n".join(lines)


def format_result_packet(packet, field_names=None):
    """格式化算法结果上报报文 (0x23) 为可读字符串。"""
    min_len = HEADER_SIZE + RESULT_HEADER_SIZE
    if len(packet) < min_len:
        return f"结果报文长度异常: {len(packet)}B, HEX={packet.hex().upper()}"

    ctrl, prog, command, sequence, data_length = struct.unpack(
        HEADER_FORMAT, packet[:HEADER_SIZE]
    )
    algo_id, result_seq, sec, usec, reserved, result_len = struct.unpack(
        RESULT_HEADER_FORMAT, packet[HEADER_SIZE:min_len]
    )
    payload = packet[min_len:min_len + result_len]

    lines = [
        "[PHM-结果上报报文]",
        f"  HEX: {packet.hex().upper()}",
        f"  消息头: ctrl=0x{ctrl:02X}, prog=0x{prog:02X}, cmd=0x{command:02X}, seq={sequence}, data_len={data_length}",
        f"  算法ID: {algo_id}, 结果序列: {result_seq}, 时间: {sec}.{usec:06d}",
        f"  结果载荷长度: {result_len}B",
    ]

    if result_len > 0 and result_len % 4 == 0:
        values = struct.unpack(f"<{result_len // 4}I", payload)
        names = field_names or [f"字段{i}" for i in range(len(values))]
        for i, value in enumerate(values):
            name = names[i] if i < len(names) else f"字段{i}"
            lines.append(f"  [{i}] {name} = {value}")
    else:
        lines.append(f"  载荷HEX: {payload.hex().upper()}")

    return "\n".join(lines)
