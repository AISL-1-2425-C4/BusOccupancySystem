if __name__ == '__main__':
    from ultralytics import YOLO

    # Load a model
    model = YOLO("best.pt")

    metrics = model.val(split='val', data="data.yaml")
    print(metrics)  

    # Validate with a custom dataset
    metrics = model.val(split='test', data="data.yaml")
    print(metrics)  