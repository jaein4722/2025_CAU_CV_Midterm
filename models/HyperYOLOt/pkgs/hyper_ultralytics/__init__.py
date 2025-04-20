# Ultralytics YOLO 🚀, AGPL-3.0 license

__version__ = '8.0.227'

from hyper_ultralytics.models import RTDETR, SAM, YOLO
from hyper_ultralytics.models.fastsam import FastSAM
from hyper_ultralytics.models.nas import NAS
from hyper_ultralytics.utils import SETTINGS as settings
from hyper_ultralytics.utils.checks import check_yolo as checks
from hyper_ultralytics.utils.downloads import download

__all__ = '__version__', 'YOLO', 'NAS', 'SAM', 'FastSAM', 'RTDETR', 'checks', 'download', 'settings'
import sys; sys.modules['ultralytics'] = sys.modules[__name__]