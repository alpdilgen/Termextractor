"""API Manager for rate limiting, caching, and API coordination."""

import asyncio
import time
import hashlib
import json
from typing import Any, Callable, Dict, List, Optional
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from loguru import logger

from .anthropic_client import AnthropicClient, APIResponse
from utils.helpers import hash_text


@dataclass
class RateLimitConfig:
    """Rate limiting configuration."""

    requests_per_minute: int = 50
    tokens_per_minute: int = 100000
    max_concurrent_requests: int = 10
    enable_rate_limiting: bool = True


@dataclass
class CacheConfig:
    """Caching configuration."""

    enabled: bool = True
    ttl_hours: int = 24
    max_size_mb: int = 500
    cache_dir: Path = field(default_factory=lambda: Path("temp/cache"))


class APIManager:
    """
    Manages API interactions with advanced features.

    Features:
    - Rate limiting (requests and tokens per minute)
    - Intelligent caching
    - Request queuing
    - Token limit management
    - Cost tracking and alerts
    - Automatic retries with backoff
    """

    def __init__(
        self,
        client: AnthropicClient,
        rate_limit_config: Optional[RateLimitConfig] = None,
        cache_config: Optional[CacheConfig] = None,
        max_cost_per_day: Optional[float] = None,
    ):
        """
        Initialize APIManager.

        Args:
            client: AnthropicClient instance
            rate_limit_config: Rate limiting configuration
            cache_config: Caching configuration
            max_cost_per_day: Maximum cost per day (USD)
        """
        self.client = client
        self.rate_limit_config = rate_limit_config or RateLimitConfig()
        self.cache_config = cache_config or CacheConfig()
        self.max_cost_per_day = max_cost_per_day

        # Create cache directory
        if self.cache_config.enabled:
            self.cache_config.cache_dir.mkdir(parents=True, exist_ok=True)

        # Rate limiting state
        self.request_timestamps: List[float] = []
        self.token_usage_timestamps: List[tuple[float, int]] = []  # (timestamp, tokens)
        self.request_semaphore = asyncio.Semaphore(
            self.rate_limit_config.max_concurrent_requests
        )

        # Cost tracking
        self.daily_costs: Dict[str, float] = {}  # date -> cost

        # Request queue
        self.queued_requests: List[Dict[str, Any]] = []

        logger.info("APIManager initialized")

    async def generate_text(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        use_cache: bool = True,
        **kwargs,
    ) -> APIResponse:
        """
        Generate text with rate limiting and caching.

        Args:
            prompt: User prompt
            system_prompt: System prompt
            use_cache: Whether to use cache
            **kwargs: Additional arguments for client

        Returns:
            APIResponse
        """
        # Check cache first
        if use_cache and self.cache_config.enabled:
            cached_response = self._get_cached_response(prompt, system_prompt)
            if cached_response:
                logger.debug("Cache hit")
                return cached_response

        # Check cost limit
        if self.max_cost_per_day:
            await self._check_cost_limit()

        # Rate limiting
        await self._apply_rate_limiting()

        # Make API call with semaphore
        async with self.request_semaphore:
            response = await self.client.generate_text(
                prompt=prompt,
                system_prompt=system_prompt,
                **kwargs,
            )

        # Update rate limiting state
        self._record_request(response.input_tokens + response.output_tokens)

        # Update cost tracking
        self._record_cost(response.cost)

        # Cache response
        if use_cache and self.cache_config.enabled:
            self._cache_response(prompt, system_prompt, response)

        return response

    async def extract_terms(
        self,
        text: str,
        source_lang: str,
        target_lang: Optional[str] = None,
        domain: Optional[str] = None,
        context: Optional[str] = None,
        use_cache: bool = True,
    ) -> Dict[str, Any]:
        """
        Extract terms with caching and rate limiting.

        Args:
            text: Source text
            source_lang: Source language
            target_lang: Target language
            domain: Domain context
            context: Additional context
            use_cache: Use cache

        Returns:
            Extraction results
        """
        # Check cache
        if use_cache and self.cache_config.enabled:
            cache_key = self._generate_cache_key(
                "extract",
                text,
                source_lang,
                target_lang or "",
                domain or "",
                context or "",
            )
            cached_result = self._get_cached_result(cache_key)
            if cached_result:
                logger.debug("Cache hit for term extraction")
                return cached_result

        # Check cost and rate limits
        if self.max_cost_per_day:
            await self._check_cost_limit()

        await self._apply_rate_limiting()

        # Make API call
        async with self.request_semaphore:
            result = await self.client.extract_terms_from_text(
                text=text,
                source_lang=source_lang,
                target_lang=target_lang,
                domain=domain,
                context=context,
            )

        # Record metrics
        if "_metadata" in result:
            total_tokens = (
                result["_metadata"]["input_tokens"]
                + result["_metadata"]["output_tokens"]
            )
            self._record_request(total_tokens)
            self._record_cost(result["_metadata"]["cost"])

        # Cache result
        if use_cache and self.cache_config.enabled:
            self._cache_result(cache_key, result)

        return result

    async def _apply_rate_limiting(self) -> None:
        """Apply rate limiting logic."""
        if not self.rate_limit_config.enable_rate_limiting:
            return

        current_time = time.time()

        # Clean old timestamps (older than 1 minute)
        cutoff_time = current_time - 60

        self.request_timestamps = [
            ts for ts in self.request_timestamps if ts > cutoff_time
        ]
        self.token_usage_timestamps = [
            (ts, tokens) for ts, tokens in self.token_usage_timestamps if ts > cutoff_time
        ]

        # Check request rate limit
        if len(self.request_timestamps) >= self.rate_limit_config.requests_per_minute:
            # Wait until oldest request is outside the window
            wait_time = 60 - (current_time - self.request_timestamps[0])
            if wait_time > 0:
                logger.info(f"Rate limit: waiting {wait_time:.2f}s")
                await asyncio.sleep(wait_time)

        # Check token rate limit
        total_tokens_last_minute = sum(
            tokens for _, tokens in self.token_usage_timestamps
        )

        if total_tokens_last_minute >= self.rate_limit_config.tokens_per_minute:
            # Wait until we're under the limit
            wait_time = 60 - (current_time - self.token_usage_timestamps[0][0])
            if wait_time > 0:
                logger.info(f"Token limit: waiting {wait_time:.2f}s")
                await asyncio.sleep(wait_time)

    def _record_request(self, tokens_used: int) -> None:
        """
        Record a request for rate limiting.

        Args:
            tokens_used: Number of tokens used
        """
        current_time = time.time()
        self.request_timestamps.append(current_time)
        self.token_usage_timestamps.append((current_time, tokens_used))

    def _record_cost(self, cost: float) -> None:
        """
        Record cost for tracking.

        Args:
            cost: Cost in USD
        """
        today = datetime.now().strftime("%Y-%m-%d")
        self.daily_costs[today] = self.daily_costs.get(today, 0.0) + cost

        logger.debug(f"Daily cost for {today}: ${self.daily_costs[today]:.4f}")

    async def _check_cost_limit(self) -> None:
        """Check if daily cost limit is exceeded."""
        if not self.max_cost_per_day:
            return

        today = datetime.now().strftime("%Y-%m-%d")
        current_cost = self.daily_costs.get(today, 0.0)

        if current_cost >= self.max_cost_per_day:
            logger.error(
                f"Daily cost limit exceeded: ${current_cost:.4f} / ${self.max_cost_per_day:.4f}"
            )
            raise Exception(
                f"Daily cost limit of ${self.max_cost_per_day} exceeded"
            )

        # Warn if approaching limit (80%)
        if current_cost >= self.max_cost_per_day * 0.8:
            logger.warning(
                f"Approaching daily cost limit: ${current_cost:.4f} / ${self.max_cost_per_day:.4f}"
            )

    def _generate_cache_key(self, *args) -> str:
        """
        Generate cache key from arguments.

        Args:
            *args: Arguments to hash

        Returns:
            Cache key (hash)
        """
        combined = "|".join(str(arg) for arg in args)
        return hash_text(combined, algorithm="sha256")

    def _get_cached_response(
        self,
        prompt: str,
        system_prompt: Optional[str],
    ) -> Optional[APIResponse]:
        """
        Get cached API response.

        Args:
            prompt: User prompt
            system_prompt: System prompt

        Returns:
            Cached response or None
        """
        cache_key = self._generate_cache_key("generate", prompt, system_prompt or "")
        cache_file = self.cache_config.cache_dir / f"{cache_key}.json"

        if not cache_file.exists():
            return None

        # Check if cache is expired
        mtime = datetime.fromtimestamp(cache_file.stat().st_mtime)
        if datetime.now() - mtime > timedelta(hours=self.cache_config.ttl_hours):
            cache_file.unlink()
            return None

        try:
            with open(cache_file, "r", encoding="utf-8") as f:
                data = json.load(f)

            return APIResponse(
                content=data["content"],
                model=data["model"],
                input_tokens=data["input_tokens"],
                output_tokens=data["output_tokens"],
                cost=data["cost"],
                latency=data["latency"],
            )

        except Exception as e:
            logger.warning(f"Failed to load cache: {e}")
            return None

    def _cache_response(
        self,
        prompt: str,
        system_prompt: Optional[str],
        response: APIResponse,
    ) -> None:
        """
        Cache API response.

        Args:
            prompt: User prompt
            system_prompt: System prompt
            response: API response
        """
        cache_key = self._generate_cache_key("generate", prompt, system_prompt or "")
        cache_file = self.cache_config.cache_dir / f"{cache_key}.json"

        try:
            data = {
                "content": response.content,
                "model": response.model,
                "input_tokens": response.input_tokens,
                "output_tokens": response.output_tokens,
                "cost": response.cost,
                "latency": response.latency,
                "cached_at": datetime.now().isoformat(),
            }

            with open(cache_file, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)

        except Exception as e:
            logger.warning(f"Failed to cache response: {e}")

    def _get_cached_result(self, cache_key: str) -> Optional[Dict[str, Any]]:
        """
        Get cached result.

        Args:
            cache_key: Cache key

        Returns:
            Cached result or None
        """
        cache_file = self.cache_config.cache_dir / f"{cache_key}.json"

        if not cache_file.exists():
            return None

        # Check expiration
        mtime = datetime.fromtimestamp(cache_file.stat().st_mtime)
        if datetime.now() - mtime > timedelta(hours=self.cache_config.ttl_hours):
            cache_file.unlink()
            return None

        try:
            with open(cache_file, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"Failed to load cached result: {e}")
            return None

    def _cache_result(self, cache_key: str, result: Dict[str, Any]) -> None:
        """
        Cache result.

        Args:
            cache_key: Cache key
            result: Result to cache
        """
        cache_file = self.cache_config.cache_dir / f"{cache_key}.json"

        try:
            with open(cache_file, "w", encoding="utf-8") as f:
                json.dump(result, f, indent=2)
        except Exception as e:
            logger.warning(f"Failed to cache result: {e}")

    def get_cost_summary(self, days: int = 7) -> Dict[str, Any]:
        """
        Get cost summary.

        Args:
            days: Number of days to include

        Returns:
            Cost summary
        """
        today = datetime.now()
        date_range = [
            (today - timedelta(days=i)).strftime("%Y-%m-%d") for i in range(days)
        ]

        costs_by_date = {
            date: self.daily_costs.get(date, 0.0) for date in date_range
        }

        total_cost = sum(costs_by_date.values())

        return {
            "date_range": date_range,
            "costs_by_date": costs_by_date,
            "total_cost": total_cost,
            "average_daily_cost": total_cost / days,
            "max_daily_limit": self.max_cost_per_day,
            "today_cost": self.daily_costs.get(today.strftime("%Y-%m-%d"), 0.0),
        }

    def clear_cache(self, older_than_hours: Optional[int] = None) -> int:
        """
        Clear cache files.

        Args:
            older_than_hours: Only clear files older than this (None = all)

        Returns:
            Number of files deleted
        """
        if not self.cache_config.enabled:
            return 0

        deleted = 0
        cutoff_time = None

        if older_than_hours:
            cutoff_time = datetime.now() - timedelta(hours=older_than_hours)

        try:
            for cache_file in self.cache_config.cache_dir.glob("*.json"):
                if cutoff_time:
                    mtime = datetime.fromtimestamp(cache_file.stat().st_mtime)
                    if mtime > cutoff_time:
                        continue

                cache_file.unlink()
                deleted += 1

            logger.info(f"Cleared {deleted} cache files")

        except Exception as e:
            logger.error(f"Error clearing cache: {e}")

        return deleted
