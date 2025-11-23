"""
FastAPI-based IDS Dashboard
---------------------------
Provides endpoints:
 - GET /logs      -> returns parsed logs as JSON
 - GET /alerts    -> returns list of detected anomalies
 - GET /dashboard -> simple HTML dashboard (Jinja2) showing recent alerts

Run with:
    pip install -r requirements.txt
    uvicorn app:app --reload

The app uses functions from `parse_logs.py` to parse logs and run detections.
"""

from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.templating import Jinja2Templates
from typing import List, Dict
import os

# Import parser and detection helpers from project
try:
    from parse_logs import parse_network_logs, detect_anomalies, ml_isolation_forest
except Exception as e:
    # Import errors will be surfaced when endpoints are hit; still define app
    parse_network_logs = None
    detect_anomalies = None
    ml_isolation_forest = None
    _import_error = e
else:
    _import_error = None

app = FastAPI(title="IDS Dashboard")

# Allow all origins for simplicity (adjust in production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

templates = Jinja2Templates(directory=os.path.join(os.path.dirname(__file__), 'templates'))


def _ensure_parser_available():
    if _import_error is not None or parse_network_logs is None:
        raise HTTPException(status_code=500, detail=f"Parser functions not available: {_import_error}")


@app.get('/logs')
def get_logs():
    """Return parsed logs as JSON list of records."""
    _ensure_parser_available()
    df = parse_network_logs()
    if df is None:
        raise HTTPException(status_code=500, detail="Failed to parse network logs")
    # Convert timestamps to ISO strings for JSON
    try:
        records = df.copy()
        if 'timestamp' in records.columns:
            records['timestamp'] = records['timestamp'].astype(str)
        return JSONResponse(records.to_dict(orient='records'))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


def _build_alerts_list(df) -> List[Dict]:
    alerts = []
    for _, row in df.iterrows():
        if not row.get('detected_anomaly', False) and not row.get('ml_detected', False):
            continue
        types = []
        if row.get('detected_anomaly', False):
            types.append('rule')
        if row.get('ml_detected', False):
            types.append('ML')
        ts = row.get('timestamp', '')
        try:
            ts_str = ts.isoformat() if hasattr(ts, 'isoformat') else str(ts)
        except Exception:
            ts_str = str(ts)
        alerts.append({
            'src_ip': row.get('src_ip', ''),
            'timestamp': ts_str,
            'type': '+'.join(types),
            'protocol': row.get('protocol', ''),
            'port': int(row.get('port', 0)) if row.get('port', None) is not None else None,
            'payload_size': int(row.get('payload_size', 0)) if row.get('payload_size', None) is not None else None,
            'is_malicious': bool(row.get('is_malicious', False)),
        })
    return alerts


@app.get('/alerts')
def get_alerts():
    """Return a JSON list of detected anomalies (rule-based and ML)."""
    _ensure_parser_available()
    df = parse_network_logs()
    if df is None:
        raise HTTPException(status_code=500, detail='Failed to parse network logs')
    df = detect_anomalies(df)
    # Run ML detection if available
    try:
        df = ml_isolation_forest(df)
    except Exception:
        # If ML not available, continue with rule-based detections only
        pass

    alerts = _build_alerts_list(df)
    return JSONResponse(alerts)


@app.get('/dashboard', response_class=HTMLResponse)
def dashboard(request: Request):
    """Render a simple dashboard page showing recent alerts."""
    _ensure_parser_available()
    df = parse_network_logs()
    if df is None:
        return HTMLResponse(content="<h3>Error: Could not parse logs</h3>", status_code=500)
    df = detect_anomalies(df)
    try:
        df = ml_isolation_forest(df)
    except Exception:
        pass
    alerts = _build_alerts_list(df)
    # Pass the most recent 100 alerts to the template
    recent = list(reversed(alerts))[:100]
    return templates.TemplateResponse('dashboard.html', {"request": request, "alerts": recent})


if __name__ == '__main__':
    # Quick import-check run (starts a uvicorn server if executed directly)
    import uvicorn
    uvicorn.run('app:app', host='0.0.0.0', port=8000, reload=True)
