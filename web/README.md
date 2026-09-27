# IRVision web app

Next.js 16 frontend for IRVision: an animated walkthrough of the processing pipeline, an
interactive playground and the evaluation results.

## Modes

| Mode | When | What it shows |
|---|---|---|
| **Demo** | `NEXT_PUBLIC_API_URL` is empty or the server is unreachable | Real results of the trained models, pre-computed into `public/demo/` by `scripts/export_web_demo.py` |
| **Live** | `NEXT_PUBLIC_API_URL` points to a running model server | Runs the pipeline on the server; also allows uploading your own images |

## Local development

```bash
npm install
cp .env.example .env.local          # optional: set NEXT_PUBLIC_API_URL=http://localhost:8000
npm run dev                          # http://localhost:3000
```

To run the model server locally (from the repository root):

```bash
.venv\Scripts\python.exe -m uvicorn irvision.api.server:app --port 8000
```

## Refreshing the demo data

After retraining or re-evaluating, regenerate the pre-computed results (from the repository root):

```bash
python scripts/export_web_demo.py
```

## Deployment

See the **Deployment** section of the main [README](../README.md).
