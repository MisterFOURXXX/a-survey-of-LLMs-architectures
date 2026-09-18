"""Resource and training monitoring callbacks."""
import time
import threading
import collections
import psutil
from transformers import TrainerCallback


class ResourceMonitor(threading.Thread):
    """Background thread that samples CPU and memory usage."""

    def __init__(self, interval: float = 1.0):
        super().__init__(daemon=True)
        self.interval = interval
        self.running = False
        self.cpu_samples = collections.deque(maxlen=1000)
        self.memory_samples = collections.deque(maxlen=1000)

    def run(self):
        self.running = True
        while self.running:
            self.cpu_samples.append(psutil.cpu_percent(interval=self.interval))
            self.memory_samples.append(psutil.virtual_memory().percent)

    def stop(self):
        self.running = False

    def get_average(self) -> tuple[float, float]:
        if not self.cpu_samples:
            return 0.0, 0.0
        return (
            sum(self.cpu_samples) / len(self.cpu_samples),
            sum(self.memory_samples) / len(self.memory_samples),
        )

    def clear(self):
        self.cpu_samples.clear()
        self.memory_samples.clear()


class EpochMonitor(TrainerCallback):
    """Tracks per-epoch time, CPU, and memory usage."""

    def __init__(self):
        self.start_time = time.time()
        self.epoch_data = []
        self.current_epoch = None
        self.monitor = ResourceMonitor(interval=1.0)

    def on_train_begin(self, args, state, control, **kwargs):
        self.monitor.start()

    def on_epoch_begin(self, args, state, control, **kwargs):
        self.epoch_start = time.time()
        self.current_epoch = state.epoch
        self.monitor.clear()

    def on_epoch_end(self, args, state, control, **kwargs):
        epoch_time = time.time() - self.epoch_start
        avg_cpu, avg_memory = self.monitor.get_average()
        self.epoch_data.append({
            "epoch": self.current_epoch,
            "time": epoch_time,
            "avg_cpu": avg_cpu,
            "avg_memory": avg_memory,
        })

    def on_train_end(self, args, state, control, **kwargs):
        self.monitor.stop()