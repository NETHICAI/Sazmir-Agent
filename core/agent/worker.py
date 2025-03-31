"""
Worker Agent - Specialized Agent for Distributed Task Execution
"""

from __future__ import annotations
import asyncio
import os
import psutil
from typing import Dict, List, Optional, Tuple
import uuid
from datetime import datetime
from pydantic import Field, validator
import numpy as np
from .base import BaseAgent, AgentMessage, AgentID, AgentConfig, AgentNetworkError

# Custom Types
TaskID = str
ResourceProfile = Dict[str, float]  # e.g. {"cpu": 1.2, "mem_gb": 4}

class WorkerConfig(AgentConfig):
    """Extended configuration for worker agents"""
    max_concurrent_tasks: int = Field(5, gt=0)
    task_timeout: int = 300  # seconds
    resource_limits: ResourceProfile = {"cpu": 2.0, "mem_gb": 8}
    task_queue_endpoint: str = "http://task-queue:8000"
    
    @validator('resource_limits')
    def validate_resources(cls, v):
        if v["cpu"] <= 0 or v["mem_gb"] <= 0:
            raise ValueError("Resource limits must be positive")
        return v

class TaskRequest(BaseModel):
    """Schema for task execution requests"""
    task_id: TaskID = Field(default_factory=lambda: f"task_{uuid.uuid4().hex[:8]}")
    payload: Dict[str, Any]
    priority: int = 1
    deadline: Optional[datetime] = None
    required_resources: ResourceProfile = {"cpu": 0.5, "mem_gb": 1}

class TaskResult(BaseModel):
    """Schema for task execution results"""
    task_id: TaskID
    success: bool
    metrics: Dict[str, float]  # e.g. {"duration": 2.3, "accuracy": 0.95}
    artifacts: List[str] = []  # Storage URIs

class WorkerMetrics(BaseModel):
    """Real-time resource utilization metrics"""
    cpu_usage: float
    mem_usage_gb: float
    active_tasks: int
    queue_size: int

class WorkerAgent(BaseAgent):
    """
    Specialized agent for executing tasks with resource management and fault tolerance
    
    Key Capabilities:
    - Dynamic task prioritization
    - Resource-aware scheduling
    - Graceful preemption
    - Failure recovery
    """
    
    def __init__(self, agent_id: AgentID):
        super().__init__(agent_id)
        self.worker_config = WorkerConfig()
        self._task_registry: Dict[TaskID, asyncio.Task] = {}
        self._resource_semaphore = asyncio.Semaphore(
            self.worker_config.max_concurrent_tasks
        )
        self._current_load = {"cpu": 0.0, "mem_gb": 0.0}

    async def _process_message(self, message: AgentMessage) -> MessagePayload:
        """Override base message processing for task handling"""
        if message.payload_type == "TaskRequest":
            task = TaskRequest(**message.payload)
            return await self._handle_task_request(task)
        return {"status": "unhandled_message_type"}

    async def _handle_task_request(self, task: TaskRequest) -> Dict[str, Any]:
        """Validate and enqueue incoming task requests"""
        if not self._check_resources(task.required_resources):
            return {"error": "insufficient_resources"}
        
        async with self._resource_semaphore:
            task_id = task.task_id
            self._task_registry[task_id] = asyncio.create_task(
                self._execute_task(task),
                name=f"Task-{task_id}"
            )
            return {"task_id": task_id, "status": "queued"}

    async def _execute_task(self, task: TaskRequest) -> None:
        """Core task execution with monitoring and fault tolerance"""
        try:
            # Update resource tracking
            self._current_load["cpu"] += task.required_resources["cpu"]
            self._current_load["mem_gb"] += task.required_resources["mem_gb"]
            
            async with asyncio.timeout(self.worker_config.task_timeout):
                # Actual task execution (subclass should override)
                result = await self._run_task_impl(task.payload)
                
                # Report results
                await self._report_to_orchestrator(
                    TaskResult(
                        task_id=task.task_id,
                        success=True,
                        metrics=result
                    )
                )
        except asyncio.TimeoutError:
            await self._handle_task_failure(task.task_id, "timeout")
        except Exception as e:
            await self._handle_task_failure(task.task_id, str(e))
        finally:
            # Release resources
            self._current_load["cpu"] -= task.required_resources["cpu"]
            self._current_load["mem_gb"] -= task.required_resources["mem_gb"]
            self._task_registry.pop(task.task_id, None)

    async def _run_task_impl(self, payload: Dict) -> Dict[str, float]:
        """To be implemented by concrete worker subclasses"""
        # Example implementation
        await asyncio.sleep(1)  # Simulate work
        return {"duration": 1.0}

    def _check_resources(self, required: ResourceProfile) -> bool:
        """Check if task requirements fit within available resources"""
        return (
            (self._current_load["cpu"] + required["cpu"]) 
            <= self.worker_config.resource_limits["cpu"]
        ) and (
            (self._current_load["mem_gb"] + required["mem_gb"])
            <= self.worker_config.resource_limits["mem_gb"]
        )

    async def _report_to_orchestrator(self, result: TaskResult) -> None:
        """Report task results to central orchestrator"""
        try:
            # TODO: Implement actual reporting logic
            pass
        except AgentNetworkError as e:
            self._logger.error(f"Failed to report task {result.task_id}: {e}")

    async def _handle_task_failure(self, task_id: TaskID, reason: str) -> None:
        """Handle task failures with retry logic"""
        self._logger.error(f"Task {task_id} failed: {reason}")
        await self._report_to_orchestrator(
            TaskResult(task_id=task_id, success=False, metrics={"error": reason})
        )

    async def _sync_state(self) -> None:
        """Override state sync to include resource metrics"""
        await super()._sync_state()
        # Collect system metrics
        self._current_state = np.array([
            self._current_load["cpu"],
            self._current_load["mem_gb"],
            len(self._task_registry),
            psutil.cpu_percent(),
            psutil.virtual_memory().percent
        ])

    async def _execute_policy(self, state: NDArray) -> NDArray:
        """Adapt resource allocation based on RL policy"""
        # TODO: Integrate with policy engine
        return np.zeros_like(state)

    def get_metrics(self) -> WorkerMetrics:
        """Return current worker metrics for monitoring"""
        return WorkerMetrics(
            cpu_usage=self._current_load["cpu"],
            mem_usage_gb=self._current_load["mem_gb"],
            active_tasks=len(self._task_registry),
            queue_size=self._message_queue.qsize()
        )

    async def shutdown(self) -> None:
        """Graceful shutdown with task cleanup"""
        await super().shutdown()
        # Cancel all pending tasks
        for task in self._task_registry.values():
            task.cancel()
        await asyncio.gather(
            *self._task_registry.values(), 
            return_exceptions=True
        )

    @classmethod
    def get_worker_metrics(cls) -> Dict[AgentID, WorkerMetrics]:
        """Get metrics for all active worker instances"""
        return {
            aid: worker.get_metrics()
            for aid, worker in cls._registry.items()
            if isinstance(worker, WorkerAgent)
        }

# Kubernetes-aware Worker Implementation
class K8sWorkerAgent(WorkerAgent):
    """Worker agent with Kubernetes integration"""
    
    def __init__(self, agent_id: AgentID):
        super().__init__(agent_id)
        self.pod_name = os.getenv("K8S_POD_NAME", "local")
        self.node_name = os.getenv("K8S_NODE_NAME", "local")
        
    async def _report_to_orchestrator(self, result: TaskResult) -> None:
        """Enhanced reporting with Kubernetes metadata"""
        result.artifacts.append(
            f"k8s://{self.node_name}/{self.pod_name}/tasks/{result.task_id}"
        )
        await super()._report_to_orchestrator(result)
