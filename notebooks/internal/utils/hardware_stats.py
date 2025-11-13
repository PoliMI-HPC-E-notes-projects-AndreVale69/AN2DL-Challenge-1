"""
Utility functions to log GPU and CPU statistics.
"""
import psutil

GPU_AVAILABLE = False
try:
    import GPUtil
    GPU_AVAILABLE = True
except ImportError:
    pass

def log_gpu_stats() -> None:
    """ Logs the GPU statistics if GPUtil is available. """
    if not GPU_AVAILABLE:
        print("GPUtil not available. Cannot log GPU stats.")
        return
    gpus = GPUtil.getGPUs()
    for gpu in gpus:
        print(f"GPU stats:")
        print(f"    - GPU ID: {gpu.id}"
              f"    - Name: {gpu.name}"
              f"    - Load: {gpu.load*100:.2f}%"
              f"    - Memory Used: {gpu.memoryUsed}MB"
              f"    - Memory Total: {gpu.memoryTotal}MB"
              f"    - Temperature: {gpu.temperature}°C")

def log_cpu_stats() -> None:
    """ Logs the CPU statistics including temperature and memory usage. """
    cores = psutil.sensors_temperatures().get('coretemp', [])
    cpu_percent = psutil.cpu_percent(interval=1)
    virtual_mem = psutil.virtual_memory()
    print(f"CPU stats:")
    print(f"    - CPU Usage: {cpu_percent}%")
    print(f"    - Total Memory: {virtual_mem.total / (1024 ** 3):.2f} GB")
    print(f"    - Available Memory: {virtual_mem.available / (1024 ** 3):.2f} GB")
    print(f"    - Used Memory: {virtual_mem.used / (1024 ** 3):.2f} GB")
    print(f"    - Memory Percentage: {virtual_mem.percent}%")
    for core in cores:
        print(f"    - CPU Core {core.label} Temperature: {core.current}°C")
