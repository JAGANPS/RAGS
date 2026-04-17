"""
Latency tracking and metrics utilities
"""
import time
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
from datetime import datetime
from collections import defaultdict
import statistics
import threading


@dataclass
class LatencyMetric:
    """Single latency measurement"""
    operation: str
    duration_ms: float
    timestamp: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class OperationMetrics:
    """Aggregated metrics for an operation"""
    operation: str
    count: int
    total_ms: float
    min_ms: float
    max_ms: float
    avg_ms: float
    p50_ms: float
    p95_ms: float
    p99_ms: float
    last_updated: datetime


class LatencyTracker:
    """Track and aggregate latency metrics"""

    def __init__(self, max_history: int = 1000):
        self.max_history = max_history
        self._metrics: Dict[str, List[LatencyMetric]] = defaultdict(list)
        self._lock = threading.Lock()

    def record(
        self,
        operation: str,
        duration_ms: float,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """Record a latency measurement"""
        metric = LatencyMetric(
            operation=operation,
            duration_ms=duration_ms,
            timestamp=datetime.now(),
            metadata=metadata or {}
        )

        with self._lock:
            self._metrics[operation].append(metric)
            # Trim old metrics
            if len(self._metrics[operation]) > self.max_history:
                self._metrics[operation] = self._metrics[operation][-self.max_history:]

    def get_metrics(self, operation: str) -> Optional[OperationMetrics]:
        """Get aggregated metrics for an operation"""
        with self._lock:
            metrics = self._metrics.get(operation, [])
            if not metrics:
                return None

            durations = [m.duration_ms for m in metrics]
            sorted_durations = sorted(durations)
            count = len(durations)

            return OperationMetrics(
                operation=operation,
                count=count,
                total_ms=sum(durations),
                min_ms=min(durations),
                max_ms=max(durations),
                avg_ms=statistics.mean(durations),
                p50_ms=sorted_durations[int(count * 0.50)] if count > 0 else 0,
                p95_ms=sorted_durations[int(count * 0.95)] if count > 1 else sorted_durations[-1],
                p99_ms=sorted_durations[int(count * 0.99)] if count > 1 else sorted_durations[-1],
                last_updated=metrics[-1].timestamp
            )

    def get_all_metrics(self) -> Dict[str, OperationMetrics]:
        """Get metrics for all tracked operations"""
        with self._lock:
            operations = list(self._metrics.keys())

        return {op: self.get_metrics(op) for op in operations if self.get_metrics(op)}

    def get_recent_metrics(self, operation: str, count: int = 10) -> List[LatencyMetric]:
        """Get recent metrics for an operation"""
        with self._lock:
            return list(self._metrics.get(operation, []))[-count:]

    def clear(self, operation: Optional[str] = None):
        """Clear metrics"""
        with self._lock:
            if operation:
                self._metrics.pop(operation, None)
            else:
                self._metrics.clear()

    def to_dict(self) -> Dict[str, Any]:
        """Export all metrics as dictionary"""
        all_metrics = self.get_all_metrics()
        return {
            op: {
                "count": m.count,
                "total_ms": round(m.total_ms, 2),
                "min_ms": round(m.min_ms, 2),
                "max_ms": round(m.max_ms, 2),
                "avg_ms": round(m.avg_ms, 2),
                "p50_ms": round(m.p50_ms, 2),
                "p95_ms": round(m.p95_ms, 2),
                "p99_ms": round(m.p99_ms, 2),
                "last_updated": m.last_updated.isoformat()
            }
            for op, m in all_metrics.items()
        }


class LatencyContext:
    """Context manager for timing operations"""

    def __init__(self, tracker: LatencyTracker, operation: str, metadata: Optional[Dict[str, Any]] = None):
        self.tracker = tracker
        self.operation = operation
        self.metadata = metadata or {}
        self.start_time: Optional[float] = None
        self.duration_ms: Optional[float] = None

    def __enter__(self):
        self.start_time = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.duration_ms = (time.perf_counter() - self.start_time) * 1000
        self.tracker.record(self.operation, self.duration_ms, self.metadata)
        return False


# Global tracker instance
_latency_tracker: Optional[LatencyTracker] = None


def get_latency_tracker() -> LatencyTracker:
    """Get or create latency tracker instance"""
    global _latency_tracker
    if _latency_tracker is None:
        _latency_tracker = LatencyTracker()
    return _latency_tracker


def track_latency(operation: str, metadata: Optional[Dict[str, Any]] = None) -> LatencyContext:
    """Create a latency tracking context"""
    return LatencyContext(get_latency_tracker(), operation, metadata)
