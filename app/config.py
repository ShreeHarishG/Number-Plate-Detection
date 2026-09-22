import torch


# Automatically use NVIDIA GPU if available
if torch.cuda.is_available():
    DEVICE = "cuda:0"
    print(f"[INFO] Using GPU: {torch.cuda.get_device_name(0)}")
else:
    DEVICE = "cpu"
    print("[WARNING] CUDA not available. Using CPU.")


# Detection settings
VEHICLE_CONFIDENCE = 0.60
PLATE_CONFIDENCE = 0.35