#!/usr/bin/env python3
# -*- coding:utf-8 -*-

__version__ = "0.3.0"

# ─── 로컬 pkgs 를 최상위 yolox 로 인식시키는 패치 ───
import sys, os
from pathlib import Path

# 이 파일이 있는 yolox/ 디렉터리 기준으로 그 부모(pkgs) 경로를 꺼내
_PKGS_DIR = str(Path(__file__).resolve().parent.parent)
# 그리고 최우선 탐색 경로로 끼워 넣습니다
if _PKGS_DIR not in sys.path:
    sys.path.insert(0, _PKGS_DIR)
# ───────────────────────────────────────────────