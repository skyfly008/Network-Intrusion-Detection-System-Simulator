import pytest
import pandas as pd

from parse_logs import detect_anomalies


def make_df(rows):
    """Helper to create DataFrame from rows (list of dicts)."""
    return pd.DataFrame(rows)


def test_detect_anomalies_payload_and_port():
    # create small DataFrame with a variety of cases
    rows = [
        {'timestamp': '2025-01-01T00:00:00', 'src_ip': '192.168.0.1', 'dst_ip': '192.168.0.2', 'protocol': 'UDP', 'port': 53, 'payload_size': 50, 'is_malicious': False},
        {'timestamp': '2025-01-01T00:01:00', 'src_ip': '192.168.0.3', 'dst_ip': '192.168.0.4', 'protocol': 'UDP', 'port': 999, 'payload_size': 2000, 'is_malicious': True},
        {'timestamp': '2025-01-01T00:02:00', 'src_ip': '192.168.0.5', 'dst_ip': '192.168.0.6', 'protocol': 'ICMP', 'port': 0, 'payload_size': 100, 'is_malicious': False},
        {'timestamp': '2025-01-01T00:03:00', 'src_ip': '192.168.0.6', 'dst_ip': '192.168.0.7', 'protocol': 'UDP', 'port': 22, 'payload_size': 100, 'is_malicious': False},
    ]

    df = make_df(rows)
    # prepare minute and packets_per_minute to satisfy detect_anomalies expectations
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['minute'] = df['timestamp'].dt.floor('min')
    df['packets_per_minute'] = 1

    out = detect_anomalies(df)
    # second row should be flagged due to large payload
    assert bool(out.loc[1, 'detected_anomaly']) is True
    # fourth row port < 1024 and protocol != 'TCP' -> flagged
    assert bool(out.loc[3, 'detected_anomaly']) is True
    # all rows meet at least one rule and should be True
    assert bool(out.loc[0, 'detected_anomaly']) is True
    assert bool(out.loc[2, 'detected_anomaly']) is True


def test_ml_isolation_forest_smoke():
    # Import sklearn or skip
    pytest.importorskip('sklearn')
    from parse_logs import ml_isolation_forest

    rows = [
        {'timestamp': '2025-01-01T00:00:00', 'src_ip': 'a', 'dst_ip': 'b', 'protocol': 'TCP', 'port': 80, 'payload_size': 100, 'is_malicious': False},
        {'timestamp': '2025-01-01T00:01:00', 'src_ip': 'c', 'dst_ip': 'd', 'protocol': 'TCP', 'port': 80, 'payload_size': 50000, 'is_malicious': True},
        {'timestamp': '2025-01-01T00:02:00', 'src_ip': 'e', 'dst_ip': 'f', 'protocol': 'TCP', 'port': 80, 'payload_size': 120, 'is_malicious': False},
        {'timestamp': '2025-01-01T00:03:00', 'src_ip': 'g', 'dst_ip': 'h', 'protocol': 'TCP', 'port': 80, 'payload_size': 20000, 'is_malicious': True},
    ]

    df = make_df(rows)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['minute'] = df['timestamp'].dt.floor('min')
    df['packets_per_minute'] = 1

    out = ml_isolation_forest(df, features=['payload_size', 'port', 'packet_rate'])
    # ml_detected column should exist
    assert 'ml_detected' in out.columns
    # ensure boolean dtype
    assert out['ml_detected'].dtype == bool
