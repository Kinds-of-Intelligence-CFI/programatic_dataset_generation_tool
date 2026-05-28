"""FastAPI app serving one dataset folder to the viewer frontend.

Route precedence (registration order matters in Starlette):

  /api/...            JSON: manifest, stimuli list, single stimulus
  /assets/{path}      dataset binary assets (traversal-safe)
  /{full_path}        the built frontend bundle, with SPA fallback to index.html

The frontend build emits its own static files under ``app/`` (configured in
``vite.config.ts``) so they never collide with the dataset's ``/assets/`` route.
"""

from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, Response

from viewer import dataset_io

_STATIC_DIR = Path(__file__).parent / "static"

_NOT_BUILT_HTML = """<!doctype html>
<html><head><meta charset="utf-8"><title>Dataset Viewer</title></head>
<body style="font-family: system-ui; padding: 2rem; color: #1c1917;">
<h1>Dataset Viewer</h1>
<p>The frontend bundle has not been built yet. Run:</p>
<pre style="background:#f5f5f4;padding:1rem;">cd viewer/frontend &amp;&amp; npm install &amp;&amp; npm run build</pre>
<p>The JSON API is already live at <code>/api/manifest</code> and
<code>/api/stimuli</code>.</p>
</body></html>
"""


def create_app(dataset_dir: str | Path) -> FastAPI:
    paths = dataset_io.resolve_dataset(dataset_dir)
    app = FastAPI(title="Dataset Viewer", version="0.1.0")

    @app.get("/api/manifest")
    def get_manifest() -> JSONResponse:
        return JSONResponse(dataset_io.load_manifest(paths))

    @app.get("/api/stimuli")
    def get_stimuli() -> JSONResponse:
        return JSONResponse(list(dataset_io.iter_stimuli(paths)))

    @app.get("/api/stimuli/{sample_id}")
    def get_stimulus(sample_id: str) -> JSONResponse:
        for record in dataset_io.iter_stimuli(paths):
            if str(record.get("sample_id")) == sample_id:
                return JSONResponse(record)
        raise HTTPException(status_code=404, detail=f"No stimulus {sample_id!r}")

    @app.get("/assets/{asset_path:path}")
    def get_asset(asset_path: str) -> FileResponse:
        resolved = dataset_io.safe_asset_path(paths, asset_path)
        if resolved is None:
            raise HTTPException(status_code=404, detail="asset not found")
        return FileResponse(resolved)

    @app.get("/{full_path:path}")
    def spa(full_path: str) -> Response:
        return _serve_frontend(full_path)

    return app


def _serve_frontend(full_path: str) -> Response:
    """Serve a built frontend file if it exists, else the SPA shell.

    Unknown paths fall back to ``index.html`` so client-side routes (e.g. a deep
    link to ``/stimuli/5``) resolve to the app shell instead of a 404.
    """
    if full_path:
        candidate = (_STATIC_DIR / full_path).resolve()
        static_root = _STATIC_DIR.resolve()
        if static_root in candidate.parents and candidate.is_file():
            return FileResponse(candidate)
    index = _STATIC_DIR / "index.html"
    if index.is_file():
        return FileResponse(index)
    return HTMLResponse(_NOT_BUILT_HTML)
