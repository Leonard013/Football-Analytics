# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Real-time football/soccer match video analytics application. Uses YOLOv8 for player and ball detection, OpenCV for image processing, and PySimpleGUI for the desktop GUI. Tracks per-team statistics: ball possession, tackles, passages, and out-of-bounds events.

## Running the Application

```bash
python Football_Analytics.py
```

Expects a video file (configured at top of `Football_Analytics.py`, currently `video/Example2.mp4`). The YOLOv8 model (`yolov8m.pt`) auto-downloads on first run.

No build step, no tests, no linting configured.

## Dependencies

`opencv-python`, `numpy`, `scipy`, `scikit-learn`, `PySimpleGUI`, `ultralytics` — no `requirements.txt` exists.

## Architecture

Four Python files with a flat structure:

- **Football_Analytics.py** — Entry point and main loop. Reads video frames, runs YOLO detection, calculates ball possession via Euclidean distance from ball to player feet, updates GUI stats.
- **GUI.py** — PySimpleGUI layout definition and window sizing.
- **Definitions.py** — `Team` data class (color, players, possession, passages, tackles, outs). Utility functions: `get_main_colors`, `team_recognizer` (classifies players by jersey color), `resize_frame`, `field_lines` (detects field boundaries using HSV masking + Pearson correlation on contour straightness).
- **Set_up.py** — Initialization: samples random frames, extracts player bounding boxes via YOLO, uses K-means clustering on jersey colors to identify the two teams.

**Data flow:** `Set_up` samples video → K-means identifies team colors → main loop per frame: YOLO detects persons/ball → `team_recognizer` assigns players to teams by color distance → ball-to-feet distance determines possession → stats accumulate → GUI updates.
