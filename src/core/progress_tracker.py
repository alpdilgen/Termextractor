"""Progress Tracker for real-time progress visualization and estimation."""

import time
from typing import Any, Callable, Dict, List, Optional
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from loguru import logger
from rich.progress import (
    Progress,
    SpinnerColumn,
    TextColumn,
    BarColumn,
    TaskProgressColumn,
    TimeRemainingColumn,
    TimeElapsedColumn,
)
from rich.console import Console


class ProgressStatus(str, Enum):
    """Progress status."""

    NOT_STARTED = "not_started"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class ProgressMetrics:
    """Metrics for progress tracking."""

    total_items: int
    completed_items: int = 0
    failed_items: int = 0
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    estimated_completion_time: Optional[float] = None
    current_rate: float = 0.0  # Items per second
    tokens_used: int = 0
    estimated_cost: float = 0.0


@dataclass
class TaskProgress:
    """Progress information for a task."""

    task_id: str
    task_name: str
    status: ProgressStatus = ProgressStatus.NOT_STARTED
    metrics: ProgressMetrics = field(default_factory=lambda: ProgressMetrics(total_items=0))
    subtasks: Dict[str, "TaskProgress"] = field(default_factory=dict)
    current_message: str = ""


class ProgressTracker:
    """
    Provides real-time progress tracking and visualization.

    Features:
    - Real-time progress bars
    - Time estimation
    - Token usage tracking
    - Cost estimation
    - Hierarchical task tracking
    - Progress callbacks
    """

    def __init__(
        self,
        enable_rich_output: bool = True,
        show_time_estimation: bool = True,
        show_token_usage: bool = True,
        show_cost_estimation: bool = True,
    ):
        """
        Initialize ProgressTracker.

        Args:
            enable_rich_output: Enable rich console output
            show_time_estimation: Show time estimation
            show_token_usage: Show token usage
            show_cost_estimation: Show cost estimation
        """
        self.enable_rich_output = enable_rich_output
        self.show_time_estimation = show_time_estimation
        self.show_token_usage = show_token_usage
        self.show_cost_estimation = show_cost_estimation

        self.tasks: Dict[str, TaskProgress] = {}
        self.console = Console() if enable_rich_output else None
        self.progress: Optional[Progress] = None
        self.progress_tasks: Dict[str, int] = {}  # Map task_id to progress bar task id

        logger.info("ProgressTracker initialized")

    def create_task(
        self,
        task_id: str,
        task_name: str,
        total_items: int,
        parent_task_id: Optional[str] = None,
    ) -> TaskProgress:
        """
        Create a new task for tracking.

        Args:
            task_id: Unique task identifier
            task_name: Human-readable task name
            total_items: Total number of items to process
            parent_task_id: Optional parent task ID for hierarchical tracking

        Returns:
            TaskProgress instance
        """
        task = TaskProgress(
            task_id=task_id,
            task_name=task_name,
            metrics=ProgressMetrics(total_items=total_items),
        )

        if parent_task_id and parent_task_id in self.tasks:
            self.tasks[parent_task_id].subtasks[task_id] = task
        else:
            self.tasks[task_id] = task

        logger.info(f"Created task: {task_name} ({total_items} items)")
        return task

    def start_task(self, task_id: str) -> None:
        """
        Start a task.

        Args:
            task_id: Task identifier
        """
        if task_id not in self.tasks:
            logger.warning(f"Task not found: {task_id}")
            return

        task = self.tasks[task_id]
        task.status = ProgressStatus.IN_PROGRESS
        task.metrics.start_time = time.time()

        logger.info(f"Started task: {task.task_name}")

    def update_progress(
        self,
        task_id: str,
        completed: Optional[int] = None,
        increment: int = 1,
        message: Optional[str] = None,
        tokens_used: Optional[int] = None,
    ) -> None:
        """
        Update task progress.

        Args:
            task_id: Task identifier
            completed: Set completed items (absolute)
            increment: Increment completed items (relative)
            message: Current status message
            tokens_used: Tokens used (cumulative)
        """
        if task_id not in self.tasks:
            logger.warning(f"Task not found: {task_id}")
            return

        task = self.tasks[task_id]

        # Update completed count
        if completed is not None:
            task.metrics.completed_items = completed
        else:
            task.metrics.completed_items += increment

        # Update message
        if message:
            task.current_message = message

        # Update tokens
        if tokens_used is not None:
            task.metrics.tokens_used = tokens_used

        # Calculate metrics
        self._calculate_metrics(task)

        # Update progress bar if using rich
        if self.progress and task_id in self.progress_tasks:
            progress_task_id = self.progress_tasks[task_id]
            self.progress.update(
                progress_task_id,
                completed=task.metrics.completed_items,
                description=f"{task.task_name}: {message or ''}",
            )

    def _calculate_metrics(self, task: TaskProgress) -> None:
        """
        Calculate progress metrics.

        Args:
            task: Task to calculate metrics for
        """
        if task.metrics.start_time is None:
            return

        elapsed_time = time.time() - task.metrics.start_time

        # Calculate rate
        if elapsed_time > 0:
            task.metrics.current_rate = task.metrics.completed_items / elapsed_time

        # Estimate completion time
        if task.metrics.current_rate > 0:
            remaining_items = task.metrics.total_items - task.metrics.completed_items
            estimated_remaining_time = remaining_items / task.metrics.current_rate
            task.metrics.estimated_completion_time = (
                time.time() + estimated_remaining_time
            )

    def complete_task(self, task_id: str, success: bool = True) -> None:
        """
        Mark task as completed.

        Args:
            task_id: Task identifier
            success: Whether task completed successfully
        """
        if task_id not in self.tasks:
            logger.warning(f"Task not found: {task_id}")
            return

        task = self.tasks[task_id]
        task.status = ProgressStatus.COMPLETED if success else ProgressStatus.FAILED
        task.metrics.end_time = time.time()

        if success:
            task.metrics.completed_items = task.metrics.total_items

        logger.info(f"Completed task: {task.task_name} (success={success})")

    def cancel_task(self, task_id: str) -> None:
        """
        Cancel a task.

        Args:
            task_id: Task identifier
        """
        if task_id not in self.tasks:
            logger.warning(f"Task not found: {task_id}")
            return

        task = self.tasks[task_id]
        task.status = ProgressStatus.CANCELLED
        task.metrics.end_time = time.time()

        logger.info(f"Cancelled task: {task.task_name}")

    def get_progress_summary(self, task_id: str) -> Dict[str, Any]:
        """
        Get progress summary for a task.

        Args:
            task_id: Task identifier

        Returns:
            Progress summary dictionary
        """
        if task_id not in self.tasks:
            return {}

        task = self.tasks[task_id]
        metrics = task.metrics

        summary = {
            "task_id": task_id,
            "task_name": task.task_name,
            "status": task.status.value,
            "total_items": metrics.total_items,
            "completed_items": metrics.completed_items,
            "failed_items": metrics.failed_items,
            "progress_percentage": (
                metrics.completed_items / metrics.total_items * 100
                if metrics.total_items > 0
                else 0
            ),
        }

        # Add timing information
        if metrics.start_time:
            elapsed = (
                time.time() - metrics.start_time
                if metrics.end_time is None
                else metrics.end_time - metrics.start_time
            )
            summary["elapsed_time"] = elapsed
            summary["elapsed_time_formatted"] = str(timedelta(seconds=int(elapsed)))

            if metrics.estimated_completion_time and self.show_time_estimation:
                remaining = metrics.estimated_completion_time - time.time()
                if remaining > 0:
                    summary["estimated_remaining"] = remaining
                    summary["estimated_remaining_formatted"] = str(
                        timedelta(seconds=int(remaining))
                    )

            summary["items_per_second"] = metrics.current_rate

        # Add token usage
        if self.show_token_usage:
            summary["tokens_used"] = metrics.tokens_used

        # Add cost estimation
        if self.show_cost_estimation:
            summary["estimated_cost"] = metrics.estimated_cost

        return summary

    def start_rich_progress(self) -> None:
        """Start rich progress bar display."""
        if not self.enable_rich_output:
            return

        self.progress = Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
            console=self.console,
        )

        self.progress.start()

    def stop_rich_progress(self) -> None:
        """Stop rich progress bar display."""
        if self.progress:
            self.progress.stop()
            self.progress = None

    def add_rich_task(
        self,
        task_id: str,
        description: str,
        total: int,
    ) -> None:
        """
        Add task to rich progress display.

        Args:
            task_id: Task identifier
            description: Task description
            total: Total items
        """
        if not self.progress:
            return

        progress_task_id = self.progress.add_task(description, total=total)
        self.progress_tasks[task_id] = progress_task_id

    def print_summary(self, task_id: str) -> None:
        """
        Print task summary.

        Args:
            task_id: Task identifier
        """
        summary = self.get_progress_summary(task_id)

        if not summary:
            return

        if self.console:
            self.console.print("\n[bold]Task Summary[/bold]")
            self.console.print(f"Task: {summary['task_name']}")
            self.console.print(f"Status: {summary['status']}")
            self.console.print(
                f"Progress: {summary['completed_items']}/{summary['total_items']} "
                f"({summary['progress_percentage']:.1f}%)"
            )

            if "elapsed_time_formatted" in summary:
                self.console.print(f"Elapsed Time: {summary['elapsed_time_formatted']}")

            if "estimated_remaining_formatted" in summary:
                self.console.print(
                    f"Estimated Remaining: {summary['estimated_remaining_formatted']}"
                )

            if "items_per_second" in summary:
                self.console.print(f"Rate: {summary['items_per_second']:.2f} items/sec")

            if self.show_token_usage:
                self.console.print(f"Tokens Used: {summary['tokens_used']:,}")

            if self.show_cost_estimation:
                self.console.print(f"Estimated Cost: ${summary['estimated_cost']:.4f}")
        else:
            # Simple text output
            print(f"\nTask: {summary['task_name']}")
            print(f"Progress: {summary['completed_items']}/{summary['total_items']}")

    def get_all_tasks_summary(self) -> List[Dict[str, Any]]:
        """
        Get summary for all tasks.

        Returns:
            List of task summaries
        """
        return [self.get_progress_summary(task_id) for task_id in self.tasks]
