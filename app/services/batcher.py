"""
Dynamic Batching Engine
Collects individual requests and processes them as batches for GPU efficiency.
"""
import asyncio
import time
from typing import List, Any, Callable
from dataclasses import dataclass, field


@dataclass
class BatchRequest:
    inputs: Any
    future: asyncio.Future
    timestamp: float = field(default_factory=time.time)


class DynamicBatcher:
    def __init__(self, max_batch_size=32, max_wait_ms=10, process_fn=None):
        self.max_batch_size = max_batch_size
        self.max_wait_ms = max_wait_ms / 1000.0
        self.process_fn = process_fn
        self.queue: List[BatchRequest] = []
        self._lock = asyncio.Lock()
        self._processing = False
        self.stats = {"total_requests": 0, "total_batches": 0, "avg_batch_size": 0}
    
    async def predict(self, inputs: Any) -> Any:
        future = asyncio.get_event_loop().create_future()
        request = BatchRequest(inputs=inputs, future=future)
        
        async with self._lock:
            self.queue.append(request)
            self.stats["total_requests"] += 1
            
            if len(self.queue) >= self.max_batch_size:
                await self._process_batch()
            elif not self._processing:
                self._processing = True
                asyncio.create_task(self._wait_and_process())
        
        return await future
    
    async def _wait_and_process(self):
        await asyncio.sleep(self.max_wait_ms)
        async with self._lock:
            if self.queue:
                await self._process_batch()
            self._processing = False
    
    async def _process_batch(self):
        batch = self.queue[:self.max_batch_size]
        self.queue = self.queue[self.max_batch_size:]
        
        all_inputs = [r.inputs for r in batch]
        results = await self.process_fn(all_inputs)
        
        for req, result in zip(batch, results):
            if not req.future.done():
                req.future.set_result(result)
        
        self.stats["total_batches"] += 1
        n = self.stats["total_batches"]
        self.stats["avg_batch_size"] = (
            self.stats["avg_batch_size"] * (n - 1) + len(batch)
        ) / n
