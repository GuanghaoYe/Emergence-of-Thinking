import torch

def get_compute_capability():
    if torch.cuda.is_available():
        device = torch.cuda.get_device_properties(0)  # Get the first GPU device
        compute_capability = (device.major, device.minor)
        return compute_capability
    else:
        raise RuntimeError("CUDA is not available")

# compute_capability = get_compute_capability()
# print(f"Compute Capability: {compute_capability[0]}.{compute_capability[1]}")