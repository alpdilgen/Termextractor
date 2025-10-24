"""Anthropic API Client for terminology extraction."""

import anthropic
import asyncio
from typing import Any, Dict, List, Optional, Union
from dataclasses import dataclass
from loguru import logger
import time

from termextractor.utils.constants import (
    ANTHROPIC_MODELS,
    DEFAULT_MODEL,
    API_TIMEOUT,
    MAX_RETRIES,
    TOKEN_LIMITS,
    COST_PER_1K_TOKENS,
)
from termextractor.utils.helpers import estimate_tokens, calculate_cost


@dataclass
class APIResponse:
    """Response from Anthropic API."""

    content: str
    model: str
    input_tokens: int
    output_tokens: int
    cost: float
    latency: float  # seconds


@dataclass
class TokenUsage:
    """Token usage tracking."""

    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    estimated_cost: float = 0.0


class AnthropicClient:
    """
    Client for interacting with Anthropic's API.

    Features:
    - Model selection from available Anthropic models
    - Token usage tracking
    - Cost estimation and monitoring
    - Rate limiting
    - Error handling with retries
    - Batch processing support
    """

    def __init__(
        self,
        api_key: str,
        model: str = DEFAULT_MODEL,
        max_tokens: int = 4096,
        temperature: float = 0.0,
        timeout: int = API_TIMEOUT,
    ):
        """
        Initialize Anthropic client.

        Args:
            api_key: Anthropic API key
            model: Model identifier
            max_tokens: Maximum tokens per request
            temperature: Sampling temperature (0-1)
            timeout: Request timeout in seconds

        Raises:
            ValueError: If model is not supported
        """
        if model not in ANTHROPIC_MODELS:
            raise ValueError(
                f"Unsupported model: {model}. "
                f"Supported models: {ANTHROPIC_MODELS}"
            )

        self.api_key = api_key
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.timeout = timeout

        # Initialize client
        self.client = anthropic.Anthropic(api_key=api_key, timeout=timeout)

        # Usage tracking
        self.total_usage = TokenUsage()
        self.request_count = 0

        logger.info(f"AnthropicClient initialized with model: {model}")

    async def generate_text(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
    ) -> APIResponse:
        """
        Generate text using Anthropic API.

        Args:
            prompt: User prompt
            system_prompt: Optional system prompt
            max_tokens: Max tokens (overrides default)
            temperature: Temperature (overrides default)

        Returns:
            APIResponse with generated content

        Raises:
            Exception: If API request fails
        """
        start_time = time.time()

        # Use defaults if not specified
        max_tokens = max_tokens or self.max_tokens
        temperature = temperature or self.temperature

        # Prepare messages
        messages = [{"role": "user", "content": prompt}]

        # Estimate input tokens
        estimated_input = estimate_tokens(prompt)
        if system_prompt:
            estimated_input += estimate_tokens(system_prompt)

        logger.debug(f"Estimated input tokens: {estimated_input}")

        try:
            # Make API call
            kwargs = {
                "model": self.model,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "messages": messages,
            }

            if system_prompt:
                kwargs["system"] = system_prompt

            response = self.client.messages.create(**kwargs)

            # Extract response content
            content = response.content[0].text

            # Get actual token usage
            input_tokens = response.usage.input_tokens
            output_tokens = response.usage.output_tokens

            # Calculate cost
            cost = calculate_cost(
                input_tokens,
                output_tokens,
                self.model,
                COST_PER_1K_TOKENS,
            )

            # Update tracking
            self._update_usage(input_tokens, output_tokens, cost)

            latency = time.time() - start_time

            logger.info(
                f"API request completed: {input_tokens} input + {output_tokens} output tokens, "
                f"${cost:.4f}, {latency:.2f}s"
            )

            return APIResponse(
                content=content,
                model=self.model,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                cost=cost,
                latency=latency,
            )

        except anthropic.APIError as e:
            logger.error(f"Anthropic API error: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error in API call: {e}")
            raise

    async def extract_terms_from_text(
        self,
        text: str,
        source_lang: str,
        target_lang: Optional[str] = None,
        domain: Optional[str] = None,
        context: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Extract terminology from text using AI.

        Args:
            text: Source text
            source_lang: Source language code
            target_lang: Target language code (None for monolingual)
            domain: Domain/subdomain context
            context: Additional context

        Returns:
            Dictionary with extracted terms and metadata
        """
        # Build extraction prompt
        system_prompt = self._build_extraction_system_prompt()
        user_prompt = self._build_extraction_user_prompt(
            text, source_lang, target_lang, domain, context
        )

        # Generate response
        response = await self.generate_text(
            prompt=user_prompt,
            system_prompt=system_prompt,
            max_tokens=self.max_tokens,
        )

        # Parse response
        try:
            import json

            # The response should be in JSON format
            result = json.loads(response.content)
            result["_metadata"] = {
                "input_tokens": response.input_tokens,
                "output_tokens": response.output_tokens,
                "cost": response.cost,
                "model": response.model,
            }
            return result

        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse API response as JSON: {e}")
            # Return raw response
            return {
                "terms": [],
                "raw_response": response.content,
                "error": "Failed to parse response",
            }

    def _build_extraction_system_prompt(self) -> str:
        """
        Build system prompt for term extraction.

        Returns:
            System prompt string
        """
        return """You are an expert terminology extraction system. Your task is to identify and extract technical terminology from text with high precision.

For each term you extract, provide:
1. The term itself (source language)
2. Translation (if target language specified)
3. Domain/subdomain classification
4. Part of speech
5. Definition or context
6. Relevance score (0-100)
7. Confidence score (0-100)
8. Frequency in the text
9. Additional metadata

Return your response as a valid JSON object with this structure:
{
  "terms": [
    {
      "term": "string",
      "translation": "string or null",
      "domain": "string",
      "subdomain": "string or null",
      "pos": "string",
      "definition": "string",
      "context": "string",
      "relevance_score": number,
      "confidence_score": number,
      "frequency": number,
      "is_compound": boolean,
      "is_abbreviation": boolean,
      "variants": ["string"],
      "related_terms": ["string"]
    }
  ],
  "domain_hierarchy": ["string"],
  "language_pair": "string",
  "statistics": {
    "total_terms": number,
    "high_relevance": number,
    "medium_relevance": number,
    "low_relevance": number
  }
}

Be precise and thorough. Only extract terms that are domain-specific and technically relevant."""

    def _build_extraction_user_prompt(
        self,
        text: str,
        source_lang: str,
        target_lang: Optional[str] = None,
        domain: Optional[str] = None,
        context: Optional[str] = None,
    ) -> str:
        """
        Build user prompt for term extraction.

        Args:
            text: Source text
            source_lang: Source language
            target_lang: Target language
            domain: Domain context
            context: Additional context

        Returns:
            User prompt string
        """
        prompt_parts = [
            f"Extract technical terminology from the following {source_lang} text."
        ]

        if target_lang:
            prompt_parts.append(
                f"Provide translations in {target_lang} where applicable."
            )
        else:
            prompt_parts.append(
                "This is monolingual extraction - suggest translations if possible."
            )

        if domain:
            prompt_parts.append(
                f"Focus on terms relevant to the domain: {domain}"
            )

        if context:
            prompt_parts.append(f"Additional context: {context}")

        prompt_parts.append("\nText to analyze:")
        prompt_parts.append(f"\n{text}\n")

        prompt_parts.append(
            "\nExtract all relevant technical terms and return as JSON."
        )

        return "\n".join(prompt_parts)

    async def classify_domain(
        self,
        text: str,
        suggested_domains: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Classify text into hierarchical domain categories.

        Args:
            text: Text to classify
            suggested_domains: Optional list of suggested domains

        Returns:
            Domain classification result
        """
        system_prompt = """You are an expert at classifying text into hierarchical domain categories.
Analyze the text and determine its domain hierarchy (up to 5 levels deep).

Return as JSON:
{
  "domain_hierarchy": ["Level1", "Level2", "Level3", ...],
  "confidence": number (0-100),
  "alternative_domains": [
    {
      "hierarchy": ["string"],
      "confidence": number
    }
  ],
  "keywords": ["string"],
  "reasoning": "string"
}"""

        user_prompt = f"Classify the domain of this text:\n\n{text}"

        if suggested_domains:
            user_prompt += f"\n\nConsider these suggested domains: {', '.join(suggested_domains)}"

        response = await self.generate_text(
            prompt=user_prompt,
            system_prompt=system_prompt,
        )

        try:
            import json

            return json.loads(response.content)
        except json.JSONDecodeError:
            return {
                "domain_hierarchy": ["General"],
                "confidence": 0,
                "error": "Failed to parse response",
            }

    def _update_usage(
        self,
        input_tokens: int,
        output_tokens: int,
        cost: float,
    ) -> None:
        """
        Update token usage tracking.

        Args:
            input_tokens: Input tokens used
            output_tokens: Output tokens used
            cost: Cost of request
        """
        self.total_usage.input_tokens += input_tokens
        self.total_usage.output_tokens += output_tokens
        self.total_usage.total_tokens += (input_tokens + output_tokens)
        self.total_usage.estimated_cost += cost
        self.request_count += 1

    def get_usage_stats(self) -> Dict[str, Any]:
        """
        Get usage statistics.

        Returns:
            Usage statistics dictionary
        """
        return {
            "model": self.model,
            "total_requests": self.request_count,
            "input_tokens": self.total_usage.input_tokens,
            "output_tokens": self.total_usage.output_tokens,
            "total_tokens": self.total_usage.total_tokens,
            "estimated_cost": self.total_usage.estimated_cost,
            "average_tokens_per_request": (
                self.total_usage.total_tokens / self.request_count
                if self.request_count > 0
                else 0
            ),
        }

    def reset_usage_stats(self) -> None:
        """Reset usage statistics."""
        self.total_usage = TokenUsage()
        self.request_count = 0
        logger.info("Usage statistics reset")

    def estimate_cost(self, text: str, num_requests: int = 1) -> Dict[str, Any]:
        """
        Estimate cost for processing text.

        Args:
            text: Text to process
            num_requests: Number of requests

        Returns:
            Cost estimation dictionary
        """
        estimated_input_tokens = estimate_tokens(text)
        # Assume output is similar length to input
        estimated_output_tokens = estimated_input_tokens

        total_input = estimated_input_tokens * num_requests
        total_output = estimated_output_tokens * num_requests

        estimated_cost = calculate_cost(
            total_input,
            total_output,
            self.model,
            COST_PER_1K_TOKENS,
        )

        return {
            "estimated_input_tokens": total_input,
            "estimated_output_tokens": total_output,
            "estimated_total_tokens": total_input + total_output,
            "estimated_cost": estimated_cost,
            "model": self.model,
            "num_requests": num_requests,
        }
