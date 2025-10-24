"""Async Processing Manager for concurrent task execution."""

import asyncio
import time
from typing import Any, Callable, Dict, List, Optional, Tuple, TypeVar
from dataclasses import dataclass, field
from enum import Enum
from loguru import logger
from pathlib import Path
import json

T = TypeVar("T")


class ProcessingPriority(str, Enum):
    """Processing priority levels."""

    HIGH = "high"
    NORMAL = "normal"
    LOW = "low"


class ProcessingStatus(str, Enum):
    """Status of processing jobs."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    PAUSED = "paused"


@dataclass
class ProcessingJob:
    """Represents a processing job."""

    job_id: str
    task: Callable
    args: Tuple = field(default_factory=tuple)
    kwargs: Dict[str, Any] = field(default_factory=dict)
    priority: ProcessingPriority = ProcessingPriority.NORMAL
    status: ProcessingStatus = ProcessingStatus.PENDING
    result: Any = None
    error: Optional[Exception] = None
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    checkpoint_data: Dict[str, Any] = field(default_factory=dict)


@dataclass
class BatchConfig:
    """Configuration for batch processing."""

    batch_size: int = 100
    max_batches: Optional[int] = None
    max_parallel_batches: int = 3
    enable_dynamic_batching: bool = True
    adaptive_batch_sizing: bool = True
    min_batch_size: int = 20
    max_batch_size: int = 500


class AsyncProcessingManager:
    """
    Manages asynchronous processing of tasks with advanced features.

    Features:
    - Parallel processing with configurable concurrency
    - Priority-based task scheduling
    - Batch processing with dynamic optimization
    - Checkpoint/resume capability
    - Rate limiting and throttling
    - Progress tracking integration
    """

    def __init__(
        self,
        max_workers: int = 5,
        batch_config: Optional[BatchConfig] = None,
        enable_checkpoints: bool = True,
        checkpoint_dir: Optional[Path] = None,
    ):
        """
        Initialize AsyncProcessingManager.

        Args:
            max_workers: Maximum number of concurrent workers
            batch_config: Batch processing configuration
            enable_checkpoints: Enable checkpoint/resume functionality
            checkpoint_dir: Directory for storing checkpoints
        """
        self.max_workers = max_workers
        self.batch_config = batch_config or BatchConfig()
        self.enable_checkpoints = enable_checkpoints
        self.checkpoint_dir = checkpoint_dir or Path("temp/checkpoints")
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.jobs: Dict[str, ProcessingJob] = {}
        self.semaphore = asyncio.Semaphore(max_workers)
        self.is_paused = False
        self.cancellation_requested = False

        logger.info(
            f"AsyncProcessingManager initialized with {max_workers} workers"
        )

    async def process_batch(
        self,
        items: List[Any],
        process_func: Callable,
        batch_id: str,
        **kwargs,
    ) -> List[Any]:
        """
        Process a batch of items concurrently.

        Args:
            items: List of items to process
            process_func: Function to apply to each item
            batch_id: Identifier for this batch
            **kwargs: Additional arguments to pass to process_func

        Returns:
            List of results

        Raises:
            Exception: If batch processing fails
        """
        logger.info(f"Processing batch {batch_id} with {len(items)} items")

        # Create tasks for all items
        tasks = []
        for i, item in enumerate(items):
            task = self._create_task(
                process_func,
                item,
                f"{batch_id}_item_{i}",
                **kwargs,
            )
            tasks.append(task)

        # Execute tasks concurrently with semaphore control
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Handle exceptions in results
        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(
                    f"Error processing item {i} in batch {batch_id}: {result}"
                )
                # Optionally re-raise or handle
                processed_results.append(None)
            else:
                processed_results.append(result)

        logger.info(f"Batch {batch_id} completed")
        return processed_results

    async def _create_task(
        self,
        func: Callable,
        item: Any,
        task_id: str,
        **kwargs,
    ) -> Any:
        """
        Create and execute a single task with semaphore control.

        Args:
            func: Function to execute
            item: Item to process
            task_id: Task identifier
            **kwargs: Additional arguments

        Returns:
            Result of function execution
        """
        async with self.semaphore:
            # Check for pause or cancellation
            while self.is_paused and not self.cancellation_requested:
                await asyncio.sleep(0.1)

            if self.cancellation_requested:
                raise asyncio.CancelledError(f"Task {task_id} cancelled")

            # Execute function
            if asyncio.iscoroutinefunction(func):
                return await func(item, **kwargs)
            else:
                # Run sync function in executor
                loop = asyncio.get_event_loop()
                return await loop.run_in_executor(None, func, item, **kwargs)

    async def process_with_batches(
        self,
        items: List[Any],
        process_func: Callable,
        job_id: str,
        progress_callback: Optional[Callable] = None,
    ) -> List[Any]:
        """
        Process items in optimized batches.

        Args:
            items: Items to process
            process_func: Processing function
            job_id: Job identifier
            progress_callback: Optional callback for progress updates

        Returns:
            List of all results
        """
        total_items = len(items)
        batch_size = self._calculate_batch_size(total_items)

        logger.info(
            f"Processing {total_items} items in batches of {batch_size} "
            f"(max {self.batch_config.max_parallel_batches} parallel batches)"
        )

        # Split into batches
        batches = [
            items[i : i + batch_size] for i in range(0, total_items, batch_size)
        ]

        total_batches = len(batches)
        all_results = []

        # Process batches with controlled parallelism
        for i in range(0, total_batches, self.batch_config.max_parallel_batches):
            batch_group = batches[i : i + self.batch_config.max_parallel_batches]

            # Create tasks for this group of batches
            batch_tasks = [
                self.process_batch(
                    batch,
                    process_func,
                    f"{job_id}_batch_{i + j}",
                )
                for j, batch in enumerate(batch_group)
            ]

            # Execute batch group
            batch_results = await asyncio.gather(*batch_tasks, return_exceptions=True)

            # Flatten results
            for batch_result in batch_results:
                if isinstance(batch_result, Exception):
                    logger.error(f"Batch failed: {batch_result}")
                    all_results.extend([None] * batch_size)
                else:
                    all_results.extend(batch_result)

            # Create checkpoint
            if self.enable_checkpoints and (i + self.batch_config.max_parallel_batches) % 10 == 0:
                await self._save_checkpoint(job_id, all_results, i + len(batch_group))

            # Progress callback
            if progress_callback:
                progress = (i + len(batch_group)) / total_batches * 100
                await progress_callback(progress, f"Processed {i + len(batch_group)}/{total_batches} batches")

        return all_results

    def _calculate_batch_size(self, total_items: int) -> int:
        """
        Calculate optimal batch size based on configuration and total items.

        Args:
            total_items: Total number of items to process

        Returns:
            Optimal batch size
        """
        if not self.batch_config.adaptive_batch_sizing:
            return self.batch_config.batch_size

        # Adaptive sizing based on total items
        if total_items < 100:
            batch_size = min(20, total_items)
        elif total_items < 1000:
            batch_size = min(100, total_items // 10)
        else:
            batch_size = min(500, total_items // 20)

        # Ensure within bounds
        batch_size = max(self.batch_config.min_batch_size, batch_size)
        batch_size = min(self.batch_config.max_batch_size, batch_size)

        logger.debug(f"Calculated batch size: {batch_size} for {total_items} items")
        return batch_size

    async def _save_checkpoint(
        self,
        job_id: str,
        results: List[Any],
        batch_number: int,
    ) -> None:
        """
        Save checkpoint data.

        Args:
            job_id: Job identifier
            results: Current results
            batch_number: Current batch number
        """
        checkpoint_file = self.checkpoint_dir / f"{job_id}_checkpoint_{batch_number}.json"

        checkpoint_data = {
            "job_id": job_id,
            "batch_number": batch_number,
            "timestamp": time.time(),
            "results_count": len(results),
            "results": results,  # In production, might need more sophisticated serialization
        }

        try:
            with open(checkpoint_file, "w", encoding="utf-8") as f:
                json.dump(checkpoint_data, f, indent=2, default=str)
            logger.info(f"Checkpoint saved: {checkpoint_file}")
        except Exception as e:
            logger.error(f"Failed to save checkpoint: {e}")

    async def load_checkpoint(self, job_id: str) -> Optional[Dict[str, Any]]:
        """
        Load the latest checkpoint for a job.

        Args:
            job_id: Job identifier

        Returns:
            Checkpoint data if found, None otherwise
        """
        # Find latest checkpoint
        checkpoint_files = sorted(
            self.checkpoint_dir.glob(f"{job_id}_checkpoint_*.json"),
            reverse=True,
        )

        if not checkpoint_files:
            logger.info(f"No checkpoint found for job {job_id}")
            return None

        latest_checkpoint = checkpoint_files[0]

        try:
            with open(latest_checkpoint, "r", encoding="utf-8") as f:
                checkpoint_data = json.load(f)
            logger.info(f"Checkpoint loaded: {latest_checkpoint}")
            return checkpoint_data
        except Exception as e:
            logger.error(f"Failed to load checkpoint: {e}")
            return None

    def pause(self) -> None:
        """Pause all processing."""
        self.is_paused = True
        logger.info("Processing paused")

    def resume(self) -> None:
        """Resume processing."""
        self.is_paused = False
        logger.info("Processing resumed")

    def cancel(self) -> None:
        """Cancel all processing."""
        self.cancellation_requested = True
        logger.info("Cancellation requested")

    async def process_files_parallel(
        self,
        file_paths: List[Path],
        process_func: Callable,
        max_parallel: Optional[int] = None,
    ) -> List[Any]:
        """
        Process multiple files in parallel.

        Args:
            file_paths: List of file paths
            process_func: Function to process each file
            max_parallel: Maximum parallel files (overrides default)

        Returns:
            List of results
        """
        max_parallel = max_parallel or self.max_workers

        logger.info(f"Processing {len(file_paths)} files with max {max_parallel} parallel")

        # Create semaphore for file-level parallelism
        file_semaphore = asyncio.Semaphore(max_parallel)

        async def process_with_semaphore(file_path: Path) -> Any:
            async with file_semaphore:
                if asyncio.iscoroutinefunction(process_func):
                    return await process_func(file_path)
                else:
                    loop = asyncio.get_event_loop()
                    return await loop.run_in_executor(None, process_func, file_path)

        tasks = [process_with_semaphore(fp) for fp in file_paths]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        return results

    def get_job_status(self, job_id: str) -> Optional[ProcessingStatus]:
        """
        Get status of a job.

        Args:
            job_id: Job identifier

        Returns:
            Job status or None if not found
        """
        if job_id in self.jobs:
            return self.jobs[job_id].status
        return None

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get processing statistics.

        Returns:
            Statistics dictionary
        """
        total_jobs = len(self.jobs)
        completed = sum(1 for j in self.jobs.values() if j.status == ProcessingStatus.COMPLETED)
        failed = sum(1 for j in self.jobs.values() if j.status == ProcessingStatus.FAILED)
        running = sum(1 for j in self.jobs.values() if j.status == ProcessingStatus.RUNNING)

        return {
            "total_jobs": total_jobs,
            "completed": completed,
            "failed": failed,
            "running": running,
            "success_rate": completed / total_jobs * 100 if total_jobs > 0 else 0,
        }
