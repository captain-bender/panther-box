# YOLOv11 Model Testing Script
from ultralytics import YOLO
import torch
import os

# Path to your dataset config and trained model weights
DATA_YAML = './dataset/version-2/data.yaml'
MODEL_WEIGHTS = './runs/train/yolo11n-panther_v2-box-v2/weights/best.pt'

def main():
    # Detect device (use CUDA if available)
    device = [0] if torch.cuda.is_available() else 'cpu'
    print(f'Using device: {device}')

    # Load the trained model
    model = YOLO(MODEL_WEIGHTS)
    print(f'Loaded model from: {MODEL_WEIGHTS}')

    # Validate/test on the test split
    print('\nRunning evaluation on test split...')
    results = model.val(
        data=DATA_YAML,
        split='test',              # Explicitly use test split
        imgsz=192,                 # Image size matching training
        batch=16,                  # Batch size for inference
        device=device,
        save=True,                 # Save validation results
        save_json=True,            # Save results as JSON
        project='runs/test',        # Output directory for test results
        name='yolo11n-panther_v2-box-v2',  # Experiment name
    )

    # Print results summary
    print('\n' + '='*50)
    print('TEST RESULTS SUMMARY')
    print('='*50)
    print(results)

    # Print key metrics
    if hasattr(results, 'box'):
        print(f"\nBox Detection Metrics:")
        print(f"  mAP50: {results.box.map50:.4f}")
        print(f"  mAP50-95: {results.box.map:.4f}")
    
    if hasattr(results, 'obb'):
        print(f"\nOriented Bounding Box Metrics:")
        print(f"  mAP50: {results.obb.map50:.4f}")
        print(f"  mAP50-95: {results.obb.map:.4f}")

    print(f"\nTest results saved to: runs/test/yolo11n-panther_v2-box-v2/")


if __name__ == '__main__':
    # Required on Windows when using multiprocessing in DataLoader
    from multiprocessing import freeze_support
    freeze_support()
    main()

