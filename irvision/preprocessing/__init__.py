"""Data loading and preprocessing.

Pipeline order (see scripts/prepare_dataset.py):
    landsat.load_scene -> alignment.verify_alignment -> masks (inside load_scene)
    -> normalization -> enhancement.apply_clahe -> patches
"""
