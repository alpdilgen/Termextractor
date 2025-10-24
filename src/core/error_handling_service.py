"""Error Handling Service for robust error recovery and reporting."""

import traceback
from typing import Any, Callable, Dict, List, Optional, Type
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from loguru import logger
import asyncio


class ErrorSeverity(str, Enum):
    """Error severity levels."""

    CRITICAL = "critical"  # System-breaking error
    HIGH = "high"  # Major functionality affected
    MEDIUM = "medium"  # Minor functionality affected
    LOW = "low"  # Negligible impact
    WARNING = "warning"  # Potential issue


class ErrorCategory(str, Enum):
    """Error categories for classification."""

    API_ERROR = "api_error"
    FILE_ERROR = "file_error"
    PARSING_ERROR = "parsing_error"
    VALIDATION_ERROR = "validation_error"
    NETWORK_ERROR = "network_error"
    TIMEOUT_ERROR = "timeout_error"
    CONFIGURATION_ERROR = "configuration_error"
    AUTHENTICATION_ERROR = "authentication_error"
    RATE_LIMIT_ERROR = "rate_limit_error"
    UNKNOWN_ERROR = "unknown_error"


@dataclass
class ErrorRecord:
    """Record of an error occurrence."""

    error_id: str
    timestamp: datetime
    category: ErrorCategory
    severity: ErrorSeverity
    exception: Exception
    context: Dict[str, Any] = field(default_factory=dict)
    stack_trace: str = ""
    recovery_attempted: bool = False
    recovery_successful: bool = False
    user_message: str = ""


class ErrorHandlingService:
    """
    Provides robust error handling and recovery mechanisms.

    Features:
    - Automatic error classification
    - Retry logic with exponential backoff
    - Error recovery strategies
    - Detailed error logging and reporting
    - Circuit breaker pattern
    - Fallback mechanisms
    """

    def __init__(
        self,
        max_retries: int = 4,
        retry_backoff_factor: float = 2.0,
        enable_circuit_breaker: bool = True,
        circuit_breaker_threshold: int = 5,
    ):
        """
        Initialize ErrorHandlingService.

        Args:
            max_retries: Maximum number of retry attempts
            retry_backoff_factor: Backoff multiplier for retries
            enable_circuit_breaker: Enable circuit breaker pattern
            circuit_breaker_threshold: Failures before circuit opens
        """
        self.max_retries = max_retries
        self.retry_backoff_factor = retry_backoff_factor
        self.enable_circuit_breaker = enable_circuit_breaker
        self.circuit_breaker_threshold = circuit_breaker_threshold

        self.error_records: List[ErrorRecord] = []
        self.circuit_breaker_state: Dict[str, Dict[str, Any]] = {}

        logger.info("ErrorHandlingService initialized")

    def classify_error(self, exception: Exception) -> tuple[ErrorCategory, ErrorSeverity]:
        """
        Classify error by category and severity.

        Args:
            exception: Exception to classify

        Returns:
            Tuple of (category, severity)
        """
        exception_type = type(exception).__name__
        exception_msg = str(exception).lower()

        # API errors
        if "api" in exception_msg or "anthropic" in exception_msg:
            if "rate limit" in exception_msg or "429" in exception_msg:
                return ErrorCategory.RATE_LIMIT_ERROR, ErrorSeverity.HIGH
            elif "authentication" in exception_msg or "401" in exception_msg:
                return ErrorCategory.AUTHENTICATION_ERROR, ErrorSeverity.CRITICAL
            else:
                return ErrorCategory.API_ERROR, ErrorSeverity.HIGH

        # Network errors
        elif any(
            err in exception_type
            for err in ["ConnectionError", "TimeoutError", "NetworkError"]
        ):
            return ErrorCategory.NETWORK_ERROR, ErrorSeverity.MEDIUM

        # File errors
        elif any(
            err in exception_type
            for err in ["FileNotFoundError", "PermissionError", "IOError"]
        ):
            return ErrorCategory.FILE_ERROR, ErrorSeverity.MEDIUM

        # Parsing errors
        elif "parse" in exception_msg or "xml" in exception_msg or "json" in exception_msg:
            return ErrorCategory.PARSING_ERROR, ErrorSeverity.MEDIUM

        # Validation errors
        elif "validation" in exception_msg or "invalid" in exception_msg:
            return ErrorCategory.VALIDATION_ERROR, ErrorSeverity.LOW

        # Default
        else:
            return ErrorCategory.UNKNOWN_ERROR, ErrorSeverity.MEDIUM

    def record_error(
        self,
        exception: Exception,
        context: Optional[Dict[str, Any]] = None,
    ) -> ErrorRecord:
        """
        Record an error occurrence.

        Args:
            exception: Exception that occurred
            context: Additional context information

        Returns:
            ErrorRecord instance
        """
        category, severity = self.classify_error(exception)

        error_record = ErrorRecord(
            error_id=f"err_{datetime.now().timestamp()}_{len(self.error_records)}",
            timestamp=datetime.now(),
            category=category,
            severity=severity,
            exception=exception,
            context=context or {},
            stack_trace=traceback.format_exc(),
            user_message=self._generate_user_message(exception, category),
        )

        self.error_records.append(error_record)

        # Log based on severity
        if severity == ErrorSeverity.CRITICAL:
            logger.critical(f"{category}: {exception}")
        elif severity == ErrorSeverity.HIGH:
            logger.error(f"{category}: {exception}")
        elif severity == ErrorSeverity.MEDIUM:
            logger.warning(f"{category}: {exception}")
        else:
            logger.info(f"{category}: {exception}")

        return error_record

    def _generate_user_message(
        self,
        exception: Exception,
        category: ErrorCategory,
    ) -> str:
        """
        Generate user-friendly error message.

        Args:
            exception: Exception that occurred
            category: Error category

        Returns:
            User-friendly message
        """
        messages = {
            ErrorCategory.API_ERROR: "An error occurred communicating with the API. Please check your connection and try again.",
            ErrorCategory.RATE_LIMIT_ERROR: "API rate limit exceeded. Please wait a moment and try again.",
            ErrorCategory.AUTHENTICATION_ERROR: "Authentication failed. Please check your API key.",
            ErrorCategory.FILE_ERROR: "Error accessing file. Please check the file path and permissions.",
            ErrorCategory.PARSING_ERROR: "Error parsing file content. The file may be corrupted or in an unsupported format.",
            ErrorCategory.NETWORK_ERROR: "Network error occurred. Please check your internet connection.",
            ErrorCategory.TIMEOUT_ERROR: "Operation timed out. Please try again.",
            ErrorCategory.VALIDATION_ERROR: "Invalid input provided. Please check your data and try again.",
            ErrorCategory.CONFIGURATION_ERROR: "Configuration error. Please check your settings.",
        }

        return messages.get(
            category,
            f"An unexpected error occurred: {str(exception)}",
        )

    async def retry_with_backoff(
        self,
        func: Callable,
        *args,
        max_retries: Optional[int] = None,
        **kwargs,
    ) -> Any:
        """
        Execute function with exponential backoff retry.

        Args:
            func: Function to execute
            *args: Positional arguments
            max_retries: Max retries (overrides default)
            **kwargs: Keyword arguments

        Returns:
            Function result

        Raises:
            Exception: If all retries fail
        """
        max_retries = max_retries or self.max_retries
        last_exception = None

        for attempt in range(max_retries + 1):
            try:
                # Check circuit breaker
                func_name = func.__name__
                if self._is_circuit_open(func_name):
                    raise Exception(f"Circuit breaker open for {func_name}")

                # Execute function
                if asyncio.iscoroutinefunction(func):
                    result = await func(*args, **kwargs)
                else:
                    result = func(*args, **kwargs)

                # Success - reset circuit breaker
                self._record_success(func_name)
                return result

            except Exception as e:
                last_exception = e
                error_record = self.record_error(e, {"attempt": attempt, "function": func.__name__})

                # Record failure for circuit breaker
                self._record_failure(func_name)

                # Check if we should retry
                if attempt < max_retries:
                    # Calculate backoff delay
                    delay = self.retry_backoff_factor ** attempt
                    logger.info(
                        f"Retry {attempt + 1}/{max_retries} after {delay}s for {func.__name__}"
                    )

                    # Check if error is retryable
                    if not self._is_retryable(error_record.category):
                        logger.warning(f"Error not retryable: {error_record.category}")
                        break

                    await asyncio.sleep(delay)
                else:
                    logger.error(f"All {max_retries} retries failed for {func.__name__}")

        # All retries failed
        raise last_exception

    def _is_retryable(self, category: ErrorCategory) -> bool:
        """
        Check if error category is retryable.

        Args:
            category: Error category

        Returns:
            True if retryable
        """
        retryable_categories = {
            ErrorCategory.NETWORK_ERROR,
            ErrorCategory.TIMEOUT_ERROR,
            ErrorCategory.API_ERROR,
            ErrorCategory.RATE_LIMIT_ERROR,
        }
        return category in retryable_categories

    def _is_circuit_open(self, circuit_name: str) -> bool:
        """
        Check if circuit breaker is open.

        Args:
            circuit_name: Name of the circuit

        Returns:
            True if circuit is open
        """
        if not self.enable_circuit_breaker:
            return False

        if circuit_name not in self.circuit_breaker_state:
            return False

        state = self.circuit_breaker_state[circuit_name]
        return state.get("failures", 0) >= self.circuit_breaker_threshold

    def _record_failure(self, circuit_name: str) -> None:
        """
        Record a failure for circuit breaker.

        Args:
            circuit_name: Name of the circuit
        """
        if not self.enable_circuit_breaker:
            return

        if circuit_name not in self.circuit_breaker_state:
            self.circuit_breaker_state[circuit_name] = {"failures": 0, "successes": 0}

        self.circuit_breaker_state[circuit_name]["failures"] += 1

        if self._is_circuit_open(circuit_name):
            logger.warning(f"Circuit breaker opened for {circuit_name}")

    def _record_success(self, circuit_name: str) -> None:
        """
        Record a success for circuit breaker.

        Args:
            circuit_name: Name of the circuit
        """
        if not self.enable_circuit_breaker:
            return

        if circuit_name in self.circuit_breaker_state:
            self.circuit_breaker_state[circuit_name]["failures"] = 0
            self.circuit_breaker_state[circuit_name]["successes"] += 1

    def reset_circuit_breaker(self, circuit_name: str) -> None:
        """
        Manually reset circuit breaker.

        Args:
            circuit_name: Name of the circuit
        """
        if circuit_name in self.circuit_breaker_state:
            self.circuit_breaker_state[circuit_name] = {"failures": 0, "successes": 0}
            logger.info(f"Circuit breaker reset for {circuit_name}")

    def get_error_statistics(self) -> Dict[str, Any]:
        """
        Get error statistics.

        Returns:
            Dictionary of statistics
        """
        total_errors = len(self.error_records)

        if total_errors == 0:
            return {"total_errors": 0}

        # Count by category
        by_category = {}
        for record in self.error_records:
            category = record.category.value
            by_category[category] = by_category.get(category, 0) + 1

        # Count by severity
        by_severity = {}
        for record in self.error_records:
            severity = record.severity.value
            by_severity[severity] = by_severity.get(severity, 0) + 1

        # Recovery rate
        recovery_attempts = sum(1 for r in self.error_records if r.recovery_attempted)
        recovery_successes = sum(1 for r in self.error_records if r.recovery_successful)
        recovery_rate = (
            recovery_successes / recovery_attempts * 100 if recovery_attempts > 0 else 0
        )

        return {
            "total_errors": total_errors,
            "by_category": by_category,
            "by_severity": by_severity,
            "recovery_attempts": recovery_attempts,
            "recovery_successes": recovery_successes,
            "recovery_rate": recovery_rate,
        }

    def get_recent_errors(self, count: int = 10) -> List[ErrorRecord]:
        """
        Get most recent errors.

        Args:
            count: Number of errors to return

        Returns:
            List of recent error records
        """
        return sorted(
            self.error_records,
            key=lambda x: x.timestamp,
            reverse=True,
        )[:count]
