"""Local web viewer for generated dataset folders.

Serves a dataset folder (manifest.json + stimuli.jsonl + assets/) over a small
FastAPI app so researchers can browse and sanity-check what they generated.
Launched via the ``viewer`` console script or ``python -m viewer <path>``.
"""
