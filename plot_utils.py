"""Safe plotting utilities for IDS-Simulator.

Provides `safe_plot_anomalies(df, outpath)` which uses a non-interactive
Matplotlib backend, writes atomically into `outpath`, and produces a
placeholder image when there are no anomalies.
"""
import os
import tempfile
from datetime import datetime

def safe_plot_anomalies(df, outpath='static/anomalies.png'):
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except Exception as e:
        raise RuntimeError(f"matplotlib is required to generate plots: {e}") from e

    os.makedirs(os.path.dirname(outpath) or '.', exist_ok=True)

    try:
        # If DataFrame is None or empty, create a placeholder image
        if df is None or getattr(df, 'empty', True):
            fig, ax = plt.subplots(figsize=(8,4))
            ax.text(0.5, 0.5, 'No anomalies detected', ha='center', va='center', fontsize=14)
            ax.set_axis_off()
        else:
            # Prefer to plot counts of detected anomalies per minute if available
            try:
                # df may be a pandas DataFrame; handle gracefully if not
                detected = df[df.get('detected_anomaly') == True] if hasattr(df, 'get') else None
                if detected is None or detected.empty:
                    fig, ax = plt.subplots(figsize=(8,4))
                    ax.text(0.5, 0.5, 'No anomalies detected', ha='center', va='center', fontsize=14)
                    ax.set_axis_off()
                else:
                    # Aggregate by minute if present, else by timestamp floor
                    if 'minute' in detected.columns:
                        counts = detected.groupby('minute').size()
                        x = counts.index
                        y = counts.values
                    else:
                        # fallback: count by timestamp string
                        counts = detected.groupby(detected['timestamp'].dt.floor('min')).size()
                        x = counts.index
                        y = counts.values
                    fig, ax = plt.subplots(figsize=(10,4))
                    ax.plot(x, y, '-o')
                    ax.set_title('Anomaly Counts Over Time')
                    ax.set_ylabel('Anomaly Count')
                    fig.autofmt_xdate()
            except Exception:
                # If grouping fails, render a simple placeholder with total count
                total = len(df[df.get('detected_anomaly') == True]) if hasattr(df, 'get') else 0
                fig, ax = plt.subplots(figsize=(8,4))
                ax.text(0.5, 0.5, f'{total} anomalies detected', ha='center', va='center', fontsize=14)
                ax.set_axis_off()

        # Save to a temp file then atomically replace
        dirpath = os.path.dirname(outpath) or '.'
        fd, tmp = tempfile.mkstemp(suffix='.png', dir=dirpath)
        os.close(fd)
        try:
            fig.savefig(tmp, bbox_inches='tight')
        finally:
            plt.close(fig)
        os.replace(tmp, outpath)
        return outpath
    except Exception as e:
        raise RuntimeError(f"Failed to generate plot: {e}") from e
