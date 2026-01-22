# config.py
# 定义特征列表，基于 dsfanet.md 的描述和 CSV 文件头

# 10个时序特征 (Temporal Features Stream)
# 通常涉及 IAT (Inter-Arrival Time) 的统计值和流持续时间
# 根据 CSV snippet 推测的列名，实际使用时请核对所有列名
TEMPORAL_FEATURES = [
    'FLOW_DURATION_MILLISECONDS',
    'SRC_TO_DST_IAT_MIN',
    'SRC_TO_DST_IAT_MAX',
    'SRC_TO_DST_IAT_AVG',
    'SRC_TO_DST_IAT_STD',
    'DST_TO_SRC_IAT_MIN',
    'DST_TO_SRC_IAT_MAX',
    'DST_TO_SRC_IAT_AVG',
    'DST_TO_SRC_IAT_STD',
    'FLOW_START_MILLISECONDS' # 作为一个示例填充至10个
]

# 43个静态特征 (Static Features Stream)
# 除去时序特征和标签外的特征
STATIC_FEATURES = [
    'IPV4_SRC_ADDR', 'IPV4_DST_ADDR', 'L4_SRC_PORT', 'L4_DST_PORT',
    'PROTOCOL', 'L7_PROTO', 'IN_BYTES', 'OUT_BYTES',
    'IN_PKTS', 'OUT_PKTS', 'TCP_FLAGS',
    'CLIENT_TCP_FLAGS', 'SERVER_TCP_FLAGS',
    'MIN_TTL', 'MAX_TTL', 'LONGEST_FLOW_PKT', 'SHORTEST_FLOW_PKT',
    'MIN_IP_PKT_LEN', 'MAX_IP_PKT_LEN',
    'SRC_TO_DST_SECOND_BYTES', 'DST_TO_SRC_SECOND_BYTES',
    'RETRANSMITTED_IN_BYTES', 'RETRANSMITTED_IN_PKTS',
    'TCP_WIN_MAX_IN', 'TCP_WIN_MAX_OUT',
    'ICMP_TYPE', 'ICMP_IPV4_TYPE',
    'DNS_QUERY_ID', 'DNS_QUERY_TYPE', 'DNS_TTL_ANSWER',
    'FTP_COMMAND_RET_CODE',
    # ... 用户需在此处补全剩余特征以达到43个 ...
]

# 标签列
LABEL_COLUMN = 'Label' # 假设 CSV 中标签列名为 Label

# 超参数
BATCH_SIZE = 64
EPOCHS = 20
LEARNING_RATE = 0.001
NUM_CLASSES = 2 # 二分类或多分类