from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse, RedirectResponse, FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi.templating import Jinja2Templates
import os
import traceback

from plot_utils import safe_plot_anomalies


def _import_parse_logs():
    try:
        import parse_logs
        return parse_logs
    except Exception as e:
        raise RuntimeError(f"Failed to import parse_logs: {e}") from e


app = FastAPI(title="IDS Dashboard (fixed)")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

os.makedirs("static", exist_ok=True)
os.makedirs("templates", exist_ok=True)

if not any(getattr(r, "path", None) in ("/static", "/static/{path:path}") for r in app.router.routes):
    app.mount("/static", StaticFiles(directory="static"), name="static")

templates = Jinja2Templates(directory="templates")


@app.get("/anomalies.png")
async def anomalies_png_redirect():
    return RedirectResponse(url="/static/anomalies.png")


@app.get("/generate-plot")
async def generate_plot():
    try:
        mod = _import_parse_logs()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    try:
        df = mod.parse_network_logs('network_logs.csv')
        df = mod.detect_anomalies(df)
        try:
            df = mod.ml_isolation_forest(df)
        except Exception:
            pass
        outpath = os.path.join('static', 'anomalies.png')
        safe_plot_anomalies(df, outpath)
        return JSONResponse({"message": "Plot generated", "path": "/static/anomalies.png"})
    except Exception as e:
        tb = traceback.format_exc()
        raise HTTPException(status_code=500, detail=f"Plot generation failed: {e}\n{tb}")


@app.get("/anomalies-file")
async def anomalies_file():
    path = os.path.join('static', 'anomalies.png')
    if os.path.exists(path):
        return FileResponse(path, media_type='image/png')
    raise HTTPException(status_code=404, detail="Anomalies file not found")


@app.get("/logs")
async def get_logs():
    try:
        mod = _import_parse_logs()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    df = mod.parse_network_logs('network_logs.csv')
    records = df.fillna('').to_dict(orient='records')
    return JSONResponse(records)


@app.get("/alerts")
async def get_alerts():
    try:
        mod = _import_parse_logs()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    df = mod.parse_network_logs('network_logs.csv')
    df = mod.detect_anomalies(df)
    alerts = []
    if 'detected_anomaly' in df.columns:
        for _, row in df[df['detected_anomaly'] == True].iterrows():
            alerts.append({
                'timestamp': str(row.get('timestamp')), 'src_ip': row.get('src_ip'), 'reason': row.get('anomaly_reason', 'rule')
            })
    return JSONResponse(alerts)


@app.get("/dashboard", response_class=HTMLResponse)
async def dashboard(request: Request):
    try:
        mod = None
        try:
            mod = _import_parse_logs()
        except Exception:
            mod = None
        path = os.path.join('static', 'anomalies.png')
        if not os.path.exists(path) and mod is not None:
            df = mod.parse_network_logs('network_logs.csv')
            df = mod.detect_anomalies(df)
            try:
                df = mod.ml_isolation_forest(df)
            except Exception:
                pass
            safe_plot_anomalies(df, path)
    except Exception:
        pass
    tpl_path = os.path.join('templates', 'dashboard.html')
    if os.path.exists(tpl_path):
        return templates.TemplateResponse('dashboard.html', {"request": request})
    html = """
    <html><head><title>IDS Dashboard</title></head><body>
    <h1>IDS Dashboard</h1>
    <p><a href='/generate-plot'>Generate Plot</a></p>
    <p><img src='/static/anomalies.png' alt='Anomalies' style='width:100%;'/></p>
    </body></html>
    """
    return HTMLResponse(content=html)
