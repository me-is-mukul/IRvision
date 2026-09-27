"""Semantic validation (PLAN.md Phase 7).

Question answered: *does the colorized image mean the same thing as the real one?*

* ``landcover``: 5 land-cover classes derived from ESA WorldCover (labels).
* ``segmenter``: DeepLabV3 (ImageNet-pretrained MobileNetV3 backbone) fine-tuned
  on real Landsat RGB to predict those classes.
* ``metrics``: IoU / mIoU / Dice / pixel agreement between two label maps.

The segmenter is run on the colorized RGB and on the true RGB; the agreement
between the two maps is the "semantic consistency" of the colorization.
"""
