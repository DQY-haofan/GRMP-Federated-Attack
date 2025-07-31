# device_manager.py
# This module provides a unified device management utility for selecting and optimizing TPU, GPU, or CPU usage. 
# It automatically detects the best available device and handles device-specific operations for machine learning workflows.

import torch
import os

try:
    import torch_xla
    import torch_xla.core.xla_model as xm
    import torch_xla.distributed.parallel_loader as pl

    TPU_AVAILABLE = True
except ImportError:
    TPU_AVAILABLE = False
    xm = None
    pl = None


class DeviceManager:
    """Unified management of TPU/GPU/CPU device selection and optimization"""

    def __init__(self):
        self.device_type, self.device = self._detect_device()
        print(f"🚀 Using device: {self.device_type}")

        # Set optimization parameters
        if self.device_type == 'cuda':
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = True

    def _detect_device(self):
        """Automatically detect the best available device"""
        # Priority: TPU > GPU > CPU

        # 1. Check for TPU
        if TPU_AVAILABLE and 'COLAB_TPU_ADDR' in os.environ:
            try:
                device = xm.xla_device()
                print("✅ TPU detected and initialized")
                return 'tpu', device
            except Exception as e:
                print(f"❌ TPU initialization failed: {e}")

        # 2. Check for GPU
        if torch.cuda.is_available():
            device = torch.device('cuda')
            gpu_name = torch.cuda.get_device_name(0)
            print(f"✅ GPU detected: {gpu_name}")
            return 'cuda', device

        # 3. Default to CPU
        print("⚠️  No accelerator found, using CPU")
        return 'cpu', torch.device('cpu')

    def get_device(self):
        """Retrieve the device object"""
        return self.device

    def is_tpu(self):
        """Check if the device is a TPU"""
        return self.device_type == 'tpu'

    def is_gpu(self):
        """Check if the device is a GPU"""
        return self.device_type == 'cuda'

    def move_to_device(self, tensor_or_model):
        """Move a tensor or model to the selected device"""
        return tensor_or_model.to(self.device)

    def optimizer_step(self, optimizer):
        """Perform an optimizer step (TPU requires special handling)"""
        if self.is_tpu():
            xm.optimizer_step(optimizer)
        else:
            optimizer.step()

    def mark_step(self):
        """Synchronization step required for TPU"""
        if self.is_tpu():
            xm.mark_step()

    def get_world_size(self):
        """Get the number of processes for distributed training"""
        if self.is_tpu():
            return xm.xrt_world_size()
        else:
            return 1

    def get_ordinal(self):
        """Get the rank of the current process"""
        if self.is_tpu():
            return xm.get_ordinal()
        else:
            return 0

    def create_parallel_loader(self, data_loader):
        """Create a parallel data loader (optimized for TPU)"""
        if self.is_tpu():
            return pl.ParallelLoader(data_loader, [self.device])
        else:
            return data_loader

    def reduce_mean(self, tensor):
        """Compute the mean across devices (for distributed training)"""
        if self.is_tpu():
            return xm.mesh_reduce('reduce_mean', tensor, lambda x: sum(x) / len(x))
        else:
            return tensor


# Global Device Manager instance
device_manager = DeviceManager()