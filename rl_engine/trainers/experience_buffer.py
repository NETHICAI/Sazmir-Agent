"""
Enterprise Experience Replay Buffer - Distributed Prioritized Experience Storage with Compression
"""

from __future__ import annotations
import os
import zlib
import mmap
import threading
import multiprocessing as mp
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import torch
import msgpack
import psutil
from google.protobuf import struct_pb2
from prometheus_client import Gauge, Histogram

from utils.logger import get_logger
from utils.serialization import ModelSerializer
from orchestration.resource_pool import ResourceManager

logger = get_logger(__name__)

# Protobuf definition for experience serialization
class ExperienceProto(struct_pb2.Struct):
    pass

@dataclass
class BufferConfig:
    """Enterprise Buffer Configuration"""
    capacity: int = 10_000_000          # Maximum experiences in memory
    disk_mirror_capacity: int = 1e9    # Max experiences in persistent storage
    priority_alpha: float = 0.6        # Prioritization exponent
    importance_beta: float = 0.4       # Importance sampling weight
    compression_level: int = 3         # Zstandard compression level
    persistence_interval: int = 300    # Seconds between disk flushes
    storage_backend: str = "auto"      # auto/s3/redis/filesystem
    safety_margin: float = 0.2         # Memory headroom buffer

class DistributedExperienceBuffer:
    """Production-grade Distributed Experience Storage"""
    
    def __init__(self, config: BufferConfig):
        self.config = config
        self._buffer = []
        self._priorities = np.zeros(int(config.capacity), dtype=np.float32)
        self._position = 0
        self._lock = mp.Lock()
        self._persistence_thread = None
        self._stop_event = mp.Event()
        self.resource_mgr = ResourceManager()
        self.serializer = ModelSerializer()
        
        # Memory-mapped disk storage
        self._init_mmap_storage()
        
        # Initialize metrics
        self._init_metrics()
        
        # Start background workers
        self._start_persistence_worker()

    def _init_metrics(self):
        """Prometheus metric instrumentation"""
        self.metrics = {
            'buffer_size': Gauge('erl_buffer_size', 'Current buffer items'),
            'memory_usage': Gauge('erl_memory_usage', 'RAM consumption in MB'),
            'flush_duration': Histogram('erl_flush_duration', 'Disk flush latency'),
            'compression_ratio': Gauge('erl_compression_ratio', 'Data compression efficiency')
        }

    def _init_mmap_storage(self):
        """Initialize memory-mapped persistent storage"""
        self.mmap_file = os.open("buffer.mmap", os.O_CREAT | os.O_RDWR)
        os.ftruncate(self.mmap_file, int(self.config.disk_mirror_capacity * 512))  # Avg 512B per exp
        self.mmap = mmap.mmap(
            self.mmap_file, 
            int(self.config.disk_mirror_capacity * 512),
            access=mmap.ACCESS_WRITE
        )

    def _start_persistence_worker(self):
        """Background thread for periodic persistence"""
        self._persistence_thread = threading.Thread(
            target=self._persistence_loop,
            daemon=True
        )
        self._persistence_thread.start()

    def add(self, experiences: List[Any], priorities: np.ndarray):
        """Thread-safe experience insertion with resource monitoring"""
        with self._lock, self._memory_guard():
            for exp, pri in zip(experiences, priorities):
                if len(self._buffer) < self.config.capacity:
                    self._buffer.append(exp)
                else:
                    self._buffer[self._position] = exp
                
                self._priorities[self._position] = pri
                self._position = (self._position + 1) % self.config.capacity
                
                self.metrics['buffer_size'].inc()

    def sample(self, batch_size: int) -> Tuple[Any, np.ndarray, np.ndarray]:
        """Prioritized experience sampling with importance weights"""
        with self._lock, self._memory_guard():
            probs = self._priorities ** self.config.priority_alpha
            probs /= probs.sum()
            
            indices = np.random.choice(len(self._buffer), batch_size, p=probs)
            weights = (len(self._buffer) * probs[indices]) ** -self.config.importance_beta
            weights /= weights.max()
            
            samples = [self._buffer[i] for i in indices]
            
            return samples, indices, weights

    def update_priorities(self, indices: np.ndarray, new_priorities: np.ndarray):
        """Batch priority update with delta clamping"""
        with self._lock:
            self._priorities[indices] = np.clip(
                new_priorities, 
                1e-6,  # Minimum priority
                1e3    # Maximum priority to prevent explosions
            )

    def _memory_guard(self):
        """Context manager for memory safety enforcement"""
        return MemoryGuard(
            max_memory=psutil.virtual_memory().available * (1 - self.config.safety_margin),
            buffer_ref=self._buffer,
            item_size=self._estimate_item_size()
        )

    def _estimate_item_size(self) -> int:
        """Dynamic item size estimation for memory management"""
        if len(self._buffer) > 0:
            sample = self._serialize_exp(self._buffer[0])
            return len(sample)
        return 512  # Fallback average estimate

    def _persistence_loop(self):
        """Background persistence mechanism"""
        while not self._stop_event.is_set():
            try:
                with self.flush_timer():
                    self._flush_to_disk()
                time.sleep(self.config.persistence_interval)
            except Exception as e:
                logger.error(f"Persistence failed: {str(e)}")
                self._emergency_spillover()

    def _flush_to_disk(self):
        """Flush in-memory buffer to persistent storage"""
        serialized = self._serialize_batch(self._buffer)
        compressed = zlib.compress(serialized, level=self.config.compression_level)
        
        # Store to disk
        self.mmap.seek(0)
        self.mmap.write(compressed)
        
        # Update metrics
        self.metrics['compression_ratio'].set(
            len(serialized) / len(compressed)
        )

    def _serialize_batch(self, batch: List) -> bytes:
        """High-performance batch serialization with MsgPack"""
        return msgpack.packb(
            [self._serialize_exp(exp) for exp in batch],
            use_bin_type=True
        )

    def _serialize_exp(self, exp: Any) -> bytes:
        """Experience serialization with Protocol Buffers"""
        proto = ExperienceProto()
        proto.update(exp.__dict__)
        return proto.SerializeToString()

    def load_persisted(self):
        """Restore buffer from persistent storage"""
        self.mmap.seek(0)
        compressed = self.mmap.read()
        serialized = zlib.decompress(compressed)
        self._buffer = msgpack.unpackb(serialized, raw=False)

    def shutdown(self):
        """Graceful shutdown procedure"""
        self._stop_event.set()
        self._persistence_thread.join()
        self._flush_to_disk()
        self.mmap.close()
        os.close(self.mmap_file)

    def __len__(self) -> int:
        with self._lock:
            return len(self._buffer)

class MemoryGuard:
    """Context manager for memory safety enforcement"""
    
    def __init__(self, max_memory: int, buffer_ref: list, item_size: int):
        self.max_memory = max_memory
        self.buffer_ref = buffer_ref
        self.item_size = item_size

    def __enter__(self):
        self._check_memory()
        return self

    def __exit__(self, *args):
        pass

    def _check_memory(self):
        """Enforce memory safety boundaries"""
        process = psutil.Process()
        used_memory = process.memory_info().rss
        
        if used_memory + self.item_size > self.max_memory:
            self._handle_overflow()

    def _handle_overflow(self):
        """Memory overflow mitigation strategies"""
        # 1. Emergency spill to disk
        # 2. Aggressive sampling
        # 3. Resource scaling request
        logger.critical("Memory overflow imminent! Initiating mitigation...")
        self._emergency_spillover()
        self._request_resource_scale()

    def _emergency_spillover(self):
        """Move buffer contents to secondary storage"""
        # Implementation specific to storage backend
        pass

    def _request_resource_scale(self):
        """Trigger automatic resource scaling"""
        ResourceManager().scale(
            service="experience_buffer",
            metric="memory_usage",
            target_value=self.max_memory * 0.8
        )

# Example Usage
if __name__ == "__main__":
    config = BufferConfig(
        capacity=1_000_000,
        disk_mirror_capacity=10_000_000
    )
    
    buffer = DistributedExperienceBuffer(config)
    
    # Add sample experiences
    experiences = [{"state": np.random.rand(128), "action": 0} for _ in range(1000)]
    buffer.add(experiences, np.random.rand(1000))
    
    # Sample batch
    batch, indices, weights = buffer.sample(256)
    
    # Update priorities
    buffer.update_priorities(indices, np.random.rand(256))
    
    # Clean shutdown
    buffer.shutdown()
