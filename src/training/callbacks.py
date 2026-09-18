"""
Trainer callbacks and runtime monitors.

Re-exports the EpochMonitor / ResourceMonitor from utils.monitoring for
convenience, and provides an EvaluationMonitor used at inference time.
"""
import collections
import threading
import time

import psutil

from ..utils.monitoring import EpochMonitor, ResourceMonitor


class _NonThreadedResourceSampler:
    """Lightweight, non-threaded CPU/memory sampler used by EvaluationMonitor."""

    def __init__(self, interval: float = 0.5):
        self.interval = interval
        self.running = False
        self.cpu_samples: collections.deque = collections.deque(maxlen=1000)
        self.memory_samples: collections.deque = collections.deque(maxlen=1000)

    def start(self):
        self.running = True
        self.cpu_samples.clear()
        self.memory_samples.clear()

    def stop(self):
        self.running = False

    def record(self):
        if self.running:
            self.cpu_samples.append(
                psutil.cpu_percent(interval=self.interval)
            )
            self.memory_samples.append(psutil.virtual_memory().percent)

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


class EvaluationMonitor:
    """
    Tracks per-query wall-clock time plus average CPU/memory during evaluation.

    Usage:
        mon = EvaluationMonitor()
        mon.start()
        ...
        mon.record_query(elapsed_seconds)
        ...
        mon.stop()
        summary = mon.get_summary()
    """

    def __init__(self, interval: float = 0.5):
        self.query_times: list[float] = []
        self.cpu_samples: list[float] = []
        self.memory_samples: list[float] = []
        self.monitor = _NonThreadedResourceSampler(interval=interval)

    def start(self):
        self.query_times = []
        self.cpu_samples = []
        self.memory_samples = []
        self.monitor.start()

    def stop(self):
        self.monitor.stop()

    def record_query(self, query_time: float):
        self.query_times.append(query_time)
        avg_cpu, avg_memory = self.monitor.get_average()
        self.cpu_samples.append(avg_cpu)
        self.memory_samples.append(avg_memory)
        self.monitor.record()

    def get_summary(self) -> dict:
        if not self.query_times:
            return {
                "avg_query_time": 0.0,
                "total_time": 0.0,
                "avg_cpu": 0.0,
                "avg_memory": 0.0,
                "num_queries": 0,
            }
        return {
            "avg_query_time": sum(self.query_times) / len(self.query_times),
            "total_time": sum(self.query_times),
            "avg_cpu": sum(self.cpu_samples) / len(self.cpu_samples)
            if self.cpu_samples
            else 0.0,
            "avg_memory": sum(self.memory_samples) / len(self.memory_samples)
            if self.memory_samples
            else 0.0,
            "num_queries": len(self.query_times),
        }


__all__ = [
    "EpochMonitor",
    "ResourceMonitor",
    "EvaluationMonitor",
]