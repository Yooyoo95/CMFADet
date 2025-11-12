# Ultralytics YOLO 🚀, AGPL-3.0 license

__version__ = "8.3.9"

import os

# Set ENV variables (place before imports)
if not os.environ.get("OMP_NUM_THREADS"):
    os.environ["OMP_NUM_THREADS"] = "1"  # default for reduced CPU utilization during training

from ultralytics.data.explorer.explorer import Explorer
from ultralytics.models import RTDETR, YOLO, YOLOWorld
from ultralytics.utils import ASSETS, SETTINGS
from ultralytics.utils.checks import check_yolo as checks
from ultralytics.utils.downloads import download

settings = SETTINGS
__all__ = (
    "__version__",
    "ASSETS",
    "YOLO",
    "YOLOWorld",
    "SAM",
    "RTDETR",
    "checks",
    "download",
    "settings",
    "Explorer",
)
