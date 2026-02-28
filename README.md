# Panther Box - YOLOv11 Object Detection

Training and testing YOLOv11 models for panther object detection with oriented bounding boxes.

## Project Structure

```
panther-box/
├── training.py          # Training script for YOLOv11n model
├── test.py             # Testing/evaluation script
├── yolo11n.pt          # Pretrained YOLOv11n weights
├── dataset/            # Training datasets
│   ├── version-1/      # Version 1 (classes: antenna, lidar)
│   └── version-2/      # Version 2 (classes: antenna, lidar)
└── runs/               # Training/testing outputs
    └── train/          # Trained model weights and results
        ├── yolo11n-panther_v1-box-v1/
        └── yolo11n-panther_v2-box-v2/
```

## Quick Start

### Training
```bash
python training.py
```
Trains YOLOv11n model on version-2 dataset. Results saved to `runs/train/`.

### Testing
```bash
python test.py
```
Evaluates the trained model on the test split. Results saved to `runs/test/`.

## Dataset

- **Classes**: antenna, lidar
- **Format**: YOLO detection format (standard bounding boxes)
- **Versions**: version-1 and version-2 available
- **Splits**: train, valid, test

## Model

**YOLOv11n** - Nano variant of YOLOv11 for object detection.
