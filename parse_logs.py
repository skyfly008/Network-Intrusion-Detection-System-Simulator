"""
parse_logs.py
----------------
Provides `parse_network_logs()` which reads `network_logs.csv` using pandas,
computes packet rate (packets per minute grouped by `src_ip`), fills NaNs,
sorts by timestamp, prints first 10 rows and summary statistics, and returns
the resulting DataFrame.

Error handling covers missing file and missing pandas installation.

Usage:
    python parse_logs.py

The function can also be imported and called from other modules.
"""

from __future__ import annotations
import sys
from typing import Optional


def parse_network_logs(filepath: str = 'network_logs.csv') -> Optional['pd.DataFrame']:
    """
    Parse `network_logs.csv` into a pandas DataFrame, add a column for
    packets per minute grouped by `src_ip`, handle NaNs by filling with 0,
    and sort by timestamp.

    Fields expected in CSV: timestamp, src_ip, dst_ip, protocol, port, payload_size, is_malicious

    Returns the processed DataFrame, or None on failure.
    """
    try:
        import pandas as pd
    except ImportError:
        print("Error: pandas is required but not installed. Install with: pip install pandas")
        return None

    try:
        df = pd.read_csv(filepath)
    except FileNotFoundError:
        print(f"Error: File '{filepath}' not found.")
        return None
    except Exception as e:
        print(f"Error reading '{filepath}': {e}")
        return None

    # Parse timestamps (coerce invalid to NaT)
    df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')

    # Sort by timestamp (NaT will appear at the end)
    df.sort_values('timestamp', inplace=True)

    # Fill NaN values with 0 where appropriate
    df.fillna(0, inplace=True)

    # Create per-minute bucket for rate calculation
    # If timestamp parsing failed for some rows, they will be 0 after fillna; handle gracefully
    try:
        df['minute'] = df['timestamp'].dt.floor('min')
    except Exception:
        # If timestamp column is not datetime (e.g., all zeros), create a dummy minute column
        df['minute'] = pd.to_datetime(df['timestamp'], errors='coerce').dt.floor('min')
        df['minute'] = df['minute'].fillna(pd.Timestamp(0))

    # Compute packets per minute grouped by src_ip and minute
    rate = df.groupby(['src_ip', 'minute']).size().reset_index(name='packets_per_minute')

    # Merge the rate back into the main DataFrame
    df = df.merge(rate, on=['src_ip', 'minute'], how='left')
    df['packets_per_minute'] = df['packets_per_minute'].fillna(0)

    # Reorder/sanitize columns: keep the required fields and the new metric
    expected_cols = ['timestamp', 'src_ip', 'dst_ip', 'protocol', 'port', 'payload_size', 'is_malicious']
    # Keep other columns if present, but ensure packets_per_minute comes last
    cols = [c for c in expected_cols if c in df.columns] + [c for c in df.columns if c not in expected_cols and c != 'packets_per_minute'] + ['packets_per_minute']
    df = df[cols]

    # Print first 10 rows
    print("\nFirst 10 rows:")
    try:
        with pd.option_context('display.max_rows', 10, 'display.max_columns', None):
            print(df.head(10).to_string(index=False))
    except Exception:
        print(df.head(10))

    # Print summary statistics
    print('\nSummary statistics (numeric columns):')
    try:
        print(df.describe())
    except Exception as e:
        print(f"Could not compute describe(): {e}")

    print('\nSummary statistics (all columns):')
    try:
        print(df.describe(include='all'))
    except Exception:
        # Some pandas versions may warn; fall back to a simple info
        print('\nColumn info:')
        print(df.info())

    return df


def _main():
    df = parse_network_logs('network_logs.csv')
    if df is None:
        sys.exit(1)


def detect_anomalies(df):
    """
    Apply simple rule-based detection to the parsed DataFrame and add a
    boolean column `detected_anomaly`.

    Rules (mark as anomaly if any are true):
    - payload_size > 1000
    - port < 1024 and protocol != 'TCP'
    - packet rate (packets_per_minute) > 50

    The function prints the total anomalies detected and returns the
    DataFrame with the new column.
    """
    try:
        import pandas as pd
    except ImportError:
        print("Warning: pandas not available for anomaly detection.")
        df['detected_anomaly'] = False
        return df

    # Ensure numeric types for comparisons
    df = df.copy()
    df['payload_size'] = pd.to_numeric(df.get('payload_size', 0), errors='coerce').fillna(0)
    df['port'] = pd.to_numeric(df.get('port', 0), errors='coerce').fillna(0).astype(int)
    df['protocol'] = df.get('protocol', '').astype(str).str.upper()
    df['packets_per_minute'] = pd.to_numeric(df.get('packets_per_minute', 0), errors='coerce').fillna(0)

    cond_payload = df['payload_size'] > 1000
    cond_port_protocol = (df['port'] < 1024) & (df['protocol'] != 'TCP')
    cond_rate = df['packets_per_minute'] > 50

    df['detected_anomaly'] = (cond_payload | cond_port_protocol | cond_rate).astype(bool)

    total = int(df['detected_anomaly'].sum())
    print(f"\nAnomaly Detection: {total} packets flagged as anomalies")

    # Optionally show some examples
    if total > 0:
        print('\nSample detected anomalies:')
        try:
            with pd.option_context('display.max_rows', 5, 'display.max_columns', None):
                print(df[df['detected_anomaly']].head(5).to_string(index=False))
        except Exception:
            print(df[df['detected_anomaly']].head(5))

    return df


def ml_isolation_forest(df, features=None, test_size=0.3, random_state=42):
    """
    Use Isolation Forest to detect anomalies using the specified features.

    - features: list of feature column names to use. If None, defaults to
      ['payload_size', 'port', 'packet_rate'] where 'packet_rate' is
      taken from 'packets_per_minute' if present.
    - Splits data with `train_test_split`, fits IsolationForest on the
      training set, predicts on the test set, computes accuracy against
      `is_malicious`, and adds a boolean column `ml_detected` to the
      returned DataFrame with predictions for the full dataset.

    Returns the DataFrame with an added `ml_detected` column, or the
    original DataFrame if scikit-learn is not available.
    """
    try:
        from sklearn.ensemble import IsolationForest
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import accuracy_score, classification_report
    except Exception:
        print("Error: scikit-learn is required for ML detection. Install with: pip install scikit-learn")
        df['ml_detected'] = False
        return df

    import pandas as pd

    # Prepare features
    if features is None:
        features = ['payload_size', 'port', 'packet_rate']

    df = df.copy()
    # Ensure packet_rate column exists (alias for packets_per_minute)
    if 'packet_rate' not in df.columns:
        if 'packets_per_minute' in df.columns:
            df['packet_rate'] = df['packets_per_minute']
        else:
            df['packet_rate'] = 0

    # Ensure numeric types
    for col in features:
        df[col] = pd.to_numeric(df.get(col, 0), errors='coerce').fillna(0)

    # Prepare labels (ground truth)
    if 'is_malicious' in df.columns:
        y = df['is_malicious'].astype(str).map({'True': 1, 'true': 1, '1': 1, 'False': 0, 'false': 0, '0': 0}).fillna(0).astype(int)
    else:
        y = pd.Series([0] * len(df))

    X = df[features].values

    # Split for evaluation
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y if y.nunique() > 1 else None
        )
    except Exception:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    # Fit Isolation Forest
    model = IsolationForest(random_state=random_state, contamination='auto')
    model.fit(X_train)

    # Predict on test set (-1 anomaly, 1 normal)
    preds = model.predict(X_test)
    y_pred = (preds == -1).astype(int)

    # Compute accuracy
    try:
        acc = accuracy_score(y_test, y_pred)
        print(f"\nML Detection (IsolationForest) accuracy: {acc:.4f}")
        print('\nClassification report:')
        print(classification_report(y_test, y_pred, zero_division=0))
    except Exception as e:
        print(f"Could not compute accuracy: {e}")

    # Predict on full dataset and add column
    full_preds = model.predict(df[features].values)
    df['ml_detected'] = (full_preds == -1)

    total_ml = int(df['ml_detected'].sum())
    print(f"Total ML-detected anomalies on full data: {total_ml}")

    return df


def alert_on_detections(df):
    """
    Print alerts for rows where `detected_anomaly` or `ml_detected` is True.

    Alert format:
      ALERT: Potential intrusion from [src_ip] at [timestamp] - Type: [rule or ML]

    At the end, prints a summary with total alerts.
    """
    # Defensive: ensure columns exist
    df = df.copy()
    if 'detected_anomaly' not in df.columns:
        df['detected_anomaly'] = False
    if 'ml_detected' not in df.columns:
        df['ml_detected'] = False

    # Normalize truthy values
    df['detected_anomaly'] = df['detected_anomaly'].astype(bool)
    df['ml_detected'] = df['ml_detected'].astype(bool)

    total_alerts = 0

    for _, row in df.iterrows():
        types = []
        if row['detected_anomaly']:
            types.append('rule')
        if row['ml_detected']:
            types.append('ML')
        if types:
            total_alerts += 1
            src = row.get('src_ip', 'unknown')
            ts = row.get('timestamp', '')
            # Format timestamp if it's a Timestamp
            try:
                ts_str = pd.to_datetime(ts).isoformat()
            except Exception:
                ts_str = str(ts)
            typ = '+'.join(types)
            print(f"ALERT: Potential intrusion from {src} at {ts_str} - Type: {typ}")

    print(f"\nTotal alerts: {total_alerts}")



if __name__ == '__main__':
    # Run parser and then detection + ML-based detection
    df = parse_network_logs()
    if df is None:
        sys.exit(1)
    df = detect_anomalies(df)
    # Run ML-based detection (Isolation Forest)
    try:
        df = ml_isolation_forest(df)
    except Exception as e:
        print(f"ML detection failed: {e}")
    # Raise alerts for detections
    try:
        alert_on_detections(df)
    except Exception as e:
        print(f"Alerting failed: {e}")
    # Exit successfully
    sys.exit(0)
