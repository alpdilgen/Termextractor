# src/api/anthropic_client.py

"""Anthropic API Client for terminology extraction."""

import anthropic
import asyncio
from typing import Any, Dict, List, Optional, Union, Tuple
from dataclasses import dataclass, field
from loguru import logger
import time
import json
import re # Ensure re is imported

# Corrected imports relative to src/
from utils.constants import (
    ANTHROPIC_MODELS,
    DEFAULT_MODEL,
    API_TIMEOUT,
    MAX_RETRIES,
    TOKEN_LIMITS,
    COST_PER_1K_TOKENS,
)
from utils.helpers import estimate_tokens, calculate_cost


@dataclass
class APIResponse:
    """Response from Anthropic API, including metrics."""
    content: str
    model: str
    input_tokens: int
    output_tokens: int
    cost: float
    latency: float # seconds


@dataclass
class TokenUsage:
    """Cumulative token usage tracking."""
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    estimated_cost: float = 0.0


class AnthropicClient:
    """
    Client for interacting with Anthropic's API, including usage tracking.
    Handles generating text and specific task prompts like term extraction.
    """

    def __init__(
        self,
        api_key: str,
        model: str = DEFAULT_MODEL,
        max_tokens: int = 4096, # Default max output tokens
        temperature: float = 0.0, # Default temperature for deterministic output
        timeout: int = API_TIMEOUT,
    ):
        """Initialize Anthropic client."""
        if not api_key:
             raise ValueError("Anthropic API key is required.")
        if model not in ANTHROPIC_MODELS:
            logger.warning(f"Model '{model}' not in known list {ANTHROPIC_MODELS}. Using default: {DEFAULT_MODEL}")
            model = DEFAULT_MODEL

        self.api_key = api_key
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.timeout = timeout

        try:
            self.client = anthropic.Anthropic(api_key=api_key, timeout=float(timeout)) # Timeout needs float
        except Exception as e:
            logger.error(f"Failed to initialize Anthropic client: {e}", exc_info=True)
            raise RuntimeError("Could not initialize Anthropic client.") from e

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
        """Generate text using Anthropic API (async wrapper)."""
        start_time = time.time()
        max_tokens_to_use = max_tokens or self.max_tokens
        temperature_to_use = temperature if temperature is not None else self.temperature

        messages = [{"role": "user", "content": prompt}]
        estimated_input = estimate_tokens(prompt)
        if system_prompt:
            estimated_input += estimate_tokens(system_prompt)
        logger.debug(f"Estimated input tokens: {estimated_input} for model {self.model}")

        try:
            kwargs = {
                "model": self.model,
                "max_tokens": max_tokens_to_use,
                "temperature": temperature_to_use,
                "messages": messages,
            }
            if system_prompt:
                kwargs["system"] = system_prompt

            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(None, lambda: self.client.messages.create(**kwargs))

            content = response.content[0].text if response.content else ""
            input_tokens = response.usage.input_tokens
            output_tokens = response.usage.output_tokens
            cost = calculate_cost(input_tokens, output_tokens, self.model, COST_PER_1K_TOKENS)
            self._update_usage(input_tokens, output_tokens, cost)
            latency = time.time() - start_time

            logger.info(f"API request completed: {input_tokens} in + {output_tokens} out tokens, ${cost:.4f}, {latency:.2f}s")

            return APIResponse(
                content=content, model=self.model, input_tokens=input_tokens,
                output_tokens=output_tokens, cost=cost, latency=latency,
            )

        except anthropic.APIError as e:
            logger.error(f"Anthropic API error: {e.status_code} - {e.message}", exc_info=True)
            raise RuntimeError(f"API Error: {e.message}") from e
        except Exception as e:
            logger.error(f"Unexpected error during API call: {e}", exc_info=True)
            raise RuntimeError("Unexpected error generating text.") from e

    async def extract_terms_from_text(
        self,
        text: str,
        source_lang: str,
        target_lang: Optional[str] = None,
        domain: Optional[str] = None,
        context: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Extract terminology from text using AI, returns parsed JSON or error dict."""
        system_prompt = self._build_extraction_system_prompt() # Uses the MODIFIED prompt now
        user_prompt = self._build_extraction_user_prompt(
            text, source_lang, target_lang, domain, context
        )

        try:
            response = await self.generate_text(
                prompt=user_prompt,
                system_prompt=system_prompt,
            )
        except Exception as api_err:
             logger.error(f"API call failed during term extraction: {api_err}", exc_info=True)
             return {"terms": [], "error": f"API call failed: {api_err}", "_metadata": {}}

        logger.debug(f"Raw API response content received:\n{response.content}")
        logger.info(f"Raw API response content (first 500 chars): {response.content[:500]}...")

        try:
            json_match = re.search(r'{.*}', response.content, re.DOTALL)
            if json_match:
                json_string = json_match.group(0)
                result = json.loads(json_string)
            else:
                 raise json.JSONDecodeError("No JSON object found in the response.", response.content, 0)

            parsed_term_count = len(result.get("terms", [])) if isinstance(result.get("terms"), list) else 0
            logger.info(f"Successfully parsed JSON response. Found {parsed_term_count} terms in 'terms' list.")

            result["_metadata"] = {
                "input_tokens": response.input_tokens, "output_tokens": response.output_tokens,
                "cost": response.cost, "model": response.model, "latency_seconds": response.latency,
            }
            return result

        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse API response as JSON: {e}")
            logger.error(f"Problematic Response Content (first 500 chars): {response.content[:500]}...")
            return {
                "terms": [], "raw_response": response.content, "error": f"Failed to parse API response: {e}",
                 "_metadata": {
                    "input_tokens": response.input_tokens, "output_tokens": response.output_tokens,
                    "cost": response.cost, "model": response.model, "latency_seconds": response.latency,
                 }
            }
        except Exception as e:
             logger.error(f"Unexpected error processing API response: {e}", exc_info=True)
             return {
                  "terms": [], "error": f"Unexpected error processing response: {e}",
                  "_metadata": {
                      "input_tokens": getattr(response, 'input_tokens', 0),
                      "output_tokens": getattr(response, 'output_tokens', 0),
                      "cost": getattr(response, 'cost', 0.0), "model": getattr(response, 'model', self.model),
                      "latency_seconds": getattr(response, 'latency', 0.0),
                  }
             }

    # --- Prompt Building Methods ---
    def _build_extraction_system_prompt(self) -> str:
        """Build system prompt for term extraction. (MODIFIED)"""
        # MODIFIED: Broadened scope, softened precision emphasis
        return """You are an expert terminology extraction system. Your task is to identify and extract key terms, concepts, entities, and specialized vocabulary relevant to the text's subject matter.

For each relevant item you extract, provide the following details:
1.  `term`: The item itself in the source language.
2.  `translation`: The translation in the target language (provide `null` if monolingual or untranslatable).
3.  `domain`: The primary domain (e.g., Medical, Technology, Legal). Use "General" if none applies.
4.  `subdomain`: A more specific subdomain if identifiable (e.g., Cardiology, Software Development). Provide `null` if not applicable.
5.  `pos`: The part of speech (e.g., NOUN, VERB, ADJ).
6.  `definition`: A concise definition of the term in the context of the domain.
7.  `context`: A short snippet from the original text showing the term in use.
8.  `relevance_score`: Your assessment of the item's relevance and importance within the domain/text (0-100). Score higher for core, specific concepts. Include items even if moderately relevant if they seem specific to the context.
9.  `confidence_score`: Your confidence in the accuracy of the extracted information (0-100).
10. `frequency`: How many times the exact term/phrase appears in the text.
11. `is_compound`: Boolean, true if the term is likely a compound word.
12. `is_abbreviation`: Boolean, true if the term is an abbreviation or acronym.
13. `variants`: A list of morphological or spelling variants found in the text.
14. `related_terms`: A list of semantically related terms found in the text.

Return your response ONLY as a single, valid JSON object enclosed in curly braces `{}`. Do not include any introductory text, explanations, or markdown formatting like ```json. The JSON object must have this structure:
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
    // ... more terms ...
  ],
  "domain_hierarchy": ["string"], // Your best guess based on the content
  "language_pair": "string (e.g., bg-en)",
  "statistics": { // Your calculation based on the terms list you generate
    "total_terms": number,
    "high_relevance": number, // Count with score >= 80
    "medium_relevance": number, // Count with score 60-79
    "low_relevance": number // Count with score < 60
  }
}

Balance precision and recall. Aim to capture important concepts even if only moderately technical. Calculate frequency based only on the provided text. Determine domain hierarchy based on the overall text content."""

    def _build_extraction_user_prompt(
        self,
        text: str,
        source_lang: str,
        target_lang: Optional[str] = None,
        domain: Optional[str] = None, # Domain string (e.g., "Medical / Cardiology")
        context: Optional[str] = None,
    ) -> str:
        """Build user prompt for term extraction."""
        # This prompt seems reasonable, leaving it as is for now.
        prompt_parts = [
            f"Extract key terms, concepts, entities, and specialized vocabulary from the following text.", # Adjusted wording slightly
            f"Source Language: {source_lang}",
        ]
        if target_lang: prompt_parts.append(f"Target Language: {target_lang}")
        else: prompt_parts.append("Target Language: None (monolingual extraction)")
        if domain: prompt_parts.append(f"Domain Hint: {domain}. Focus on items relevant to this domain.")
        else: prompt_parts.append("Domain Hint: None provided. Determine domain from text.")
        if context: prompt_parts.append(f"Additional Context: {context}")

        max_prompt_chars = 150000 # Adjust if needed based on model limits
        if len(text) > max_prompt_chars:
             logger.warning(f"Input text length ({len(text)}) exceeds limit ({max_prompt_chars}). Truncating.")
             text_to_send = text[:max_prompt_chars] + "\n[...TEXT TRUNCATED...]"
        else:
             text_to_send = text

        prompt_parts.append("\n--- TEXT TO ANALYZE START ---")
        prompt_parts.append(text_to_send)
        prompt_parts.append("--- TEXT TO ANALYZE END ---")
        prompt_parts.append("\nPlease provide the extracted terminology strictly in the specified JSON format.")
        return "\n".join(prompt_parts)

    # --- Domain Classification Method (Example) ---
    async def classify_domain(
        self,
        text: str,
        suggested_domains: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Classify text into hierarchical domain categories using AI."""
        # ... (Implementation seems okay, ensure prompt asks for JSON only) ...
        system_prompt = """You are an expert text classifier... Format output ONLY as JSON: { ... }""" # Truncated for brevity
        user_prompt = f"Classify the domain hierarchy of the following text:\n\n--- TEXT START ---\n{text[:5000]}...\n--- TEXT END ---"
        # ... (rest of implementation) ...
        return {"domain_hierarchy": ["General"], "confidence": 0, "error": "Not fully implemented"} # Placeholder

    # --- Usage Tracking Methods ---
    def _update_usage(self, input_tokens: int, output_tokens: int, cost: float) -> None:
        """Update cumulative token usage and cost."""
        self.total_usage.input_tokens += input_tokens
        self.total_usage.output_tokens += output_tokens
        self.total_usage.total_tokens += (input_tokens + output_tokens)
        self.total_usage.estimated_cost += cost
        self.request_count += 1

    def get_usage_stats(self) -> Dict[str, Any]:
        """Get current usage statistics for this client instance."""
        avg_tokens = (self.total_usage.total_tokens / self.request_count
                      if self.request_count > 0 else 0)
        return {
            "model": self.model, "total_requests": self.request_count,
            "input_tokens": self.total_usage.input_tokens, "output_tokens": self.total_usage.output_tokens,
            "total_tokens": self.total_usage.total_tokens, "estimated_cost": self.total_usage.estimated_cost,
            "average_tokens_per_request": avg_tokens,
        }

    def reset_usage_stats(self) -> None:
        """Reset usage statistics."""
        self.total_usage = TokenUsage()
        self.request_count = 0
        logger.info("AnthropicClient usage statistics reset.")
