# YOLOv11 Model Validation Script for Report-Tests
from ultralytics import YOLO
import torch
import json
from datetime import datetime

# Path to your dataset config and trained model weights
DATA_YAML = './dataset/report-tests/data-validation.yaml'
MODEL_WEIGHTS = './runs/train/yolo11n-panther_v2-box-v2/weights/best.pt'

def main():
    # Detect device (use CUDA if available)
    device = [0] if torch.cuda.is_available() else 'cpu'
    print(f'Using device: {device}')

    # Load the trained model
    model = YOLO(MODEL_WEIGHTS)
    print(f'Loaded model from: {MODEL_WEIGHTS}')

    # Validate on the report-tests test split
    print('\nRunning validation on report-tests dataset...')
    results = model.val(
        data=DATA_YAML,
        split='test',              # Explicitly use test split
        imgsz=192,                 # Image size matching training
        batch=16,                  # Batch size for inference
        device=device,
        save=True,                 # Save validation results
        save_json=True,            # Save results as JSON
        project='runs/test',        # Output directory for test results
        name='yolo11n-panther_v2-box-v2-report-tests-validation',  # Experiment name
    )

    # Extract metrics
    metrics = {
        "timestamp": datetime.now().isoformat(),
        "model": MODEL_WEIGHTS,
        "dataset": "report-tests",
        "total_images": 40,
        "image_size": 192,
        "batch_size": 16,
        "device": str(device),
        "performance_metrics": {}
    }

    # Print results summary
    print('\n' + '='*50)
    print('VALIDATION RESULTS SUMMARY - REPORT-TESTS')
    print('='*50)
    print(results)

    # Extract and store key metrics
    if hasattr(results, 'box'):
        print(f"\nDetection Metrics:")
        print(f"  mAP50: {results.box.map50:.4f}")
        print(f"  mAP50-95: {results.box.map:.4f}")
        
        metrics["performance_metrics"]["mAP50"] = float(results.box.map50)
        metrics["performance_metrics"]["mAP50-95"] = float(results.box.map)
        
        # Additional metrics if available
        if hasattr(results.box, 'maps'):
            metrics["performance_metrics"]["maps"] = [float(m) for m in results.box.maps]
    
    # Store class-wise metrics if available
    if hasattr(results, 'box') and hasattr(results.box, 'map_per_class'):
        class_names = results.names
        class_metrics = {}
        for class_id, class_name in class_names.items():
            if hasattr(results.box, 'map_per_class'):
                class_metrics[class_name] = float(results.box.map_per_class[class_id])
        if class_metrics:
            metrics["performance_metrics"]["map_per_class"] = class_metrics

    # Save metrics to JSON file
    json_path = 'runs/test/yolo11n-panther_v2-box-v2-report-tests-validation/metrics.json'
    with open(json_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    
    print(f"\n✓ Metrics saved to: {json_path}")
    print("\nMetrics JSON content:")
    print(json.dumps(metrics, indent=2))


if __name__ == '__main__':
    # Required on Windows when using multiprocessing in DataLoader
    from multiprocessing import freeze_support
    freeze_support()
    main()
