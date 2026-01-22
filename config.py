# config.py
# 定义特征列表，基于 dsfanet.md 的描述和 CSV 文件头

# 1. TEMPORAL FEATURES (Based on IAT and Duration stats in NetFlow v3)
# These represent time-domain behaviors.
TEMPORAL_FEATURES = [
    'FLOW_DURATION_MILLISECONDS',
    'DURATION_IN',
    'DURATION_OUT',
    'SRC_TO_DST_IAT_MIN',
    'SRC_TO_DST_IAT_MAX',
    'SRC_TO_DST_IAT_AVG',
    'SRC_TO_DST_IAT_STDDEV',
    'DST_TO_SRC_IAT_MIN',
    'DST_TO_SRC_IAT_MAX',
    'DST_TO_SRC_IAT_AVG',
    'DST_TO_SRC_IAT_STDDEV',
    'FLOW_START_MILLISECONDS',
    'FLOW_END_MILLISECONDS'
]

# 2. STATIC FEATURES (Header fields, Counts, Byte stats)
# 除去时序特征和标签外的特征
STATIC_FEATURES = [
    # Identifiers (Optional: often dropped to prevent overfitting to specific IPs)
    # 'IPV4_SRC_ADDR', 'IPV4_DST_ADDR', 
    'L4_SRC_PORT', 
    'L4_DST_PORT',
    
    # Prorocols
    'PROTOCOL', 
    'L7_PROTO', 
    
    # Volumetrics
    'IN_BYTES', 'OUT_BYTES',
    'IN_PKTS', 'OUT_PKTS', 
    
    # Flags
    'TCP_FLAGS', 'CLIENT_TCP_FLAGS', 'SERVER_TCP_FLAGS',
    
    # Packet Size Stats
    'MIN_TTL', 'MAX_TTL', 
    'LONGEST_FLOW_PKT', 'SHORTEST_FLOW_PKT',
    'MIN_IP_PKT_LEN', 'MAX_IP_PKT_LEN',
    
    # Rates and Retransmission
    'SRC_TO_DST_SECOND_BYTES', 'DST_TO_SRC_SECOND_BYTES',
    'RETRANSMITTED_IN_BYTES', 'RETRANSMITTED_IN_PKTS',
    'RETRANSMITTED_OUT_BYTES', 'RETRANSMITTED_OUT_PKTS',
    'SRC_TO_DST_AVG_THROUGHPUT', 'DST_TO_SRC_AVG_THROUGHPUT',
    
    # Histograms (Packet size buckets)
    'NUM_PKTS_UP_TO_128_BYTES',
    'NUM_PKTS_128_TO_256_BYTES',
    'NUM_PKTS_256_TO_512_BYTES',
    'NUM_PKTS_512_TO_1024_BYTES',
    'NUM_PKTS_1024_TO_1514_BYTES',
    
    # Initial Windows & Context
    'TCP_WIN_MAX_IN', 'TCP_WIN_MAX_OUT',
    'ICMP_TYPE', 'ICMP_IPV4_TYPE',
    'DNS_QUERY_ID', 'DNS_QUERY_TYPE', 'DNS_TTL_ANSWER',
    'FTP_COMMAND_RET_CODE'
]

# 假设 CSV 的最后一列是标签，或者根据实际列名指定 ('Label', 'Attack', 'class' 等)
LABEL_COLUMN = 'Label' 

# 超参数
BATCH_SIZE = 64
EPOCHS = 10
LEARNING_RATE = 0.001
NUM_CLASSES = 2