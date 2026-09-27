"""Object detection (PLAN.md §4C): pre-trained YOLOv8n-OBB (DOTA aerial imagery).

There is no object ground truth for our Landsat scenes, so detection is used the same way
as segmentation: detect on the colorized RGB and on the true RGB, and measure how
consistent the two sets of detections are (precision / recall / F1 of colorized vs true).
"""
