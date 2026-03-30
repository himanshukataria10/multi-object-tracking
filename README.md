# Multi-Object Detection and Tracking

## Overview
This project detects and tracks multiple objects in a video using YOLOv8 and DeepSORT.

## Technologies Used
- YOLOv8
- DeepSORT
- OpenCV

## Installation
pip install ultralytics opencv-python numpy deep-sort-realtime

## How to Run
python main.py

## Input
- input.mp4 (video file)

## Output
- output.mp4 (video with bounding boxes and unique IDs)

## Model Used
YOLOv8n + DeepSORT

## Features
- Detects multiple objects
- Assigns unique IDs
- Tracks objects across frames

## Limitations
- ID switching may occur in crowded scenes
- Performance depends on video quality