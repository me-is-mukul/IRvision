"""HTTP API for the web frontend.

    uvicorn irvision.api.server:app --port 8000

``payload.py`` turns a ``PipelineResult`` into a JSON-friendly dict (display-ready
images, stage timings, metrics). The same builder is used by the API and by
``scripts/export_web_demo.py``, which pre-renders results for the static demo mode.
"""
