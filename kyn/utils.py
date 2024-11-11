import torch
from typing import Optional


def validate_device(device_str: Optional[str] = None) -> torch.device:
    """
    Validates and returns a torch device across platforms (CUDA, MPS, CPU).

    Args:
        device_str: Optional device string ('cuda', 'cuda:0', 'mps', 'cpu').
                   If None, selects best available device.

    Returns:
        torch.device: Valid torch device

    Raises:
        ValueError: If the requested device is not available
    """
    if device_str is None:
        # Auto-select best available device
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    device_str = device_str.lower().strip()

    # Validate CUDA devices
    if device_str.startswith("cuda"):
        if not torch.cuda.is_available():
            raise ValueError(f"CUDA is not available. Requested device: {device_str}")

        # Validate specific GPU if index provided
        if ":" in device_str:
            device_idx = int(device_str.split(":")[1])
            if device_idx >= torch.cuda.device_count():
                raise ValueError(
                    f"GPU {device_idx} not found. Available CUDA devices: {torch.cuda.device_count()}"
                )

    # Validate MPS (Apple Silicon) device
    elif device_str == "mps":
        if not hasattr(torch.backends, "mps") or not torch.backends.mps.is_available():
            raise ValueError(
                "MPS device not available. Ensure you're using macOS 12.3+ "
                "and running on Apple Silicon."
            )

    # Validate CPU
    elif device_str != "cpu":
        raise ValueError(
            f"Unknown device: {device_str}. Valid options are: 'cuda', 'cuda:N', 'mps', 'cpu'"
        )

    try:
        device = torch.device(device_str)
        return device
    except RuntimeError as e:
        raise ValueError(f"Error creating device '{device_str}': {str(e)}")
