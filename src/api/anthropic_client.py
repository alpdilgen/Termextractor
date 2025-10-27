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
# Ensure constants are loaded correctly
try:
    from utils.constants import (
        ANTHROPIC_MODELS,
        DEFAULT_MODEL,
        API_TIMEOUT,
        MAX_RETRIES,
        TOKEN_LIMITS,
        COST_PER_1K_TOKENS,
    )
    from utils.helpers import estimate_tokens, calculate_cost
    constants_available = True
except ImportError:
    logger.error("Could not import constants or helpers from utils. Using fallback values.")
    constants_available = False
    # Define minimal fallbacks if needed, though this indicates a bigger issue
    ANTHROPIC_MODELS = ["claude-3-5-sonnet-20240620"] # Example fallback
    DEFAULT_MODEL = ANTHROPIC_MODELS[0]
    API_TIMEOUT = 300
    MAX_RETRIES = 4
    TOKEN_LIMITS = {DEFAULT_MODEL: 200000}
    COST_PER_1K_TOKENS = {DEFAULT_MODEL: {"input": 0.003, "output": 0.015}}
    def estimate_tokens(text: str) -> int: return len(text) // 4
    def calculate_cost(in_tok: int, out_tok: int, model: str, costs: dict) -> float: return 0.0


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
        # Ensure model is valid, fallback to default if not found
        effective_model = model
        if model not in ANTHROPIC_MODELS:
            logger.warning(f"Model '{model}' not in known list {ANTHROPIC_MODELS}. Using default: {DEFAULT_MODEL}")
            effective_model = DEFAULT_MODEL

        self.api_key = api_key
        self.model = effective_model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.timeout = timeout

        try:
            self.client = anthropic.Anthropic(api_key=api_key, timeout=float(timeout))
        except Exception as e:
            logger.error(f"Failed to initialize Anthropic client: {e}", exc_info=True)
            raise RuntimeError("Could not initialize Anthropic client.") from e

        self.total_usage = TokenUsage()
        self.request_count = 0
        logger.info(f"AnthropicClient initialized with model: {self.model}")

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
            # Ensure constants are available before calculating cost
            cost = calculate_cost(input_tokens, output_tokens, self.model, COST_PER_1K_TOKENS) if constants_available else 0.0
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
            response = await self.generate_text( # This makes the actual API call
                prompt=user_prompt,
                system_prompt=system_prompt,
            )
        except Exception as api_err:
             logger.error(f"API call failed during term extraction: {api_err}", exc_info=True)
             # Return a dictionary indicating failure, including potential cost if available
             return {"terms": [], "error": f"API call failed: {api_err}", "_metadata": self.get_usage_stats()}


        logger.debug(f"Raw API response content received:\n{response.content}")
        logger.info(f"Raw API response content (first 500 chars): {response.content[:500]}...")

        try:
            # Attempt to find JSON block if response contains extra text
            json_match = re.search(r'{.*}', response.content, re.DOTALL)
            if json_match:
                json_string = json_match.group(0)
                result = json.loads(json_string)
            else:
                 # Check if the content itself might be the JSON (without extra text)
                 try:
                      result = json.loads(response.content)
                 except json.JSONDecodeError:
                      # If direct parsing also fails, raise the original error logic
                      raise json.JSONDecodeError("No valid JSON object found in the response.", response.content, 0)

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
                 "_metadata": { # Include metadata even on parse failure
                    "input_tokens": response.input_tokens, "output_tokens": response.output_tokens,
                    "cost": response.cost, "model": response.model, "latency_seconds": response.latency,
                 }
            }
        except Exception as e: # Catch other potential errors during processing
             logger.error(f"Unexpected error processing API response: {e}", exc_info=True)
             return {
                  "terms": [], "error": f"Unexpected error processing response: {e}",
                  "_metadata": { # Include metadata if available from response
                      "input_tokens": getattr(response, 'input_tokens', 0),
                      "output_tokens": getattr(response, 'output_tokens', 0),
                      "cost": getattr(response, 'cost', 0.0), "model": getattr(response, 'model', self.model),
                      "latency_seconds": getattr(response, 'latency', 0.0),
                  }
             }

    # --- Prompt Building Methods ---

    # ***** MODIFIED SYSTEM PROMPT *****
    def _build_extraction_system_prompt(self) -> str:
        """Build system prompt for term extraction, inspired by v8 and broadened scope."""
        return """You are an expert terminology extraction system specialized in identifying key terms, concepts, and domain-specific vocabulary from various texts. Your goal is to provide a comprehensive list suitable for creating glossaries or termbases.

Analyze the provided text carefully based on the source language, target language (if specified), and domain hint (if provided). For each relevant item you extract, provide the following details:
1.  `term`: The term/phrase in the source language exactly as it appears or its base form.
2.  `translation`: The appropriate translation in the target language, considering the domain. Provide `null` if monolingual extraction or if no suitable translation is found.
3.  `domain`: The primary domain (e.g., Medical, Technology, Legal, Education). Use "General" if none clearly applies.
4.  `subdomain`: A more specific subdomain if identifiable (e.g., Cardiology, Software Development, Children's Literature). Provide `null` if not applicable.
5.  `pos`: The part of speech (e.g., NOUN, VERB, ADJ, PHRASE).
6.  `definition`: A concise definition, explanation, or clarification of the term, especially within the context of the domain.
7.  `context`: A short, representative snippet from the original text showing the term in use.
8.  `relevance_score`: Your assessment (0-100) of how important and specific the term is to the text's core subject matter and domain. Prioritize terms that are central to understanding the text's specialized content. Include terms even if only moderately relevant if they seem distinct from common language.
9.  `confidence_score`: Your confidence (0-100) in the accuracy of the extracted information (term identification, translation, definition).
10. `frequency`: How many times the exact term/phrase (case-insensitive recommended for counting) appears in the text.
11. `is_compound`: Boolean, true if the source term is likely a compound word (especially relevant for languages like German).
12. `is_abbreviation`: Boolean, true if the source term is an abbreviation or acronym.
13. `variants`: A list of morphological or spelling variations found in the text (e.g., plurals, different capitalizations).
14. `related_terms`: A list of semantically related terms also found in the text (synonyms, antonyms, hyponyms/hypernyms if identifiable).

Your response MUST be ONLY a single, valid JSON object enclosed in curly braces `{}`. Do not include any text before or after the JSON object. Do not use markdown formatting like ```json. The JSON structure MUST be exactly as follows:
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
    // ... potentially many more term objects ...
  ],
  "domain_hierarchy": ["string"], // Your best hierarchical classification based on the content
  "language_pair": "string (e.g., bg-en)",
  "statistics": { // Your calculation based *only* on the terms list you generate
    "total_terms": number, // Count of all terms in the list above
    "high_relevance": number, // Count where relevance_score >= 80
    "medium_relevance": number, // Count where relevance_score is 60-79
    "low_relevance": number // Count where relevance_score < 60
  }
}

Be thorough but discerning. Identify terms specific to the domain or subject matter, distinguishing them from general vocabulary. Aim for a balance between capturing important concepts (recall) and ensuring accuracy (precision). If the text is non-technical (like a children's story), extract vocabulary that might be key for that specific context or learning level, even if not strictly 'technical'. Calculate frequency based only on the provided text. Determine the domain hierarchy based on the overall text content."""
    # ***** END OF MODIFIED SYSTEM PROMPT *****

    def _build_extraction_user_prompt(
        self,
        text: str,
        source_lang: str,
        target_lang: Optional[str] = None,
        domain: Optional[str] = None, # Domain string (e.g., "Medical / Cardiology")
        context: Optional[str] = None,
    ) -> str:
        """Build user prompt for term extraction."""
        # This prompt seems reasonable, making only minor wording adjustments
        prompt_parts = [
            f"Extract key terms, concepts, entities, and specialized vocabulary from the following text.",
            f"Source Language: {source_lang}",
        ]
        if target_lang: prompt_parts.append(f"Target Language: {target_lang}")
        else: prompt_parts.append("Target Language: None (monolingual extraction)")
        if domain: prompt_parts.append(f"Domain Hint: {domain}. Focus on items relevant to this domain.")
        else: prompt_parts.append("Domain Hint: None provided. Determine domain from text if possible.")
        if context: prompt_parts.append(f"Additional Context: {context}")

        # Basic truncation if text is extremely long (adjust limit as needed)
        max_prompt_chars = 180000 # Increased limit slightly, still well within typical context windows
        text_to_send = text
        if len(text) > max_prompt_chars:
             logger.warning(f"Input text length ({len(text)}) exceeds limit ({max_prompt_chars}). Truncating.")
             # Keep start and end for context
             keep_chars = max_prompt_chars // 2
             text_to_send = f"{text[:keep_chars]}\n\n[...TEXT TRUNCATED...]\n\n{text[-keep_chars:]}"

        prompt_parts.append("\n--- TEXT TO ANALYZE START ---")
        prompt_parts.append(text_to_send)
        prompt_parts.append("--- TEXT TO ANALYZE END ---")
        prompt_parts.append("\nPlease provide the extracted items strictly in the specified JSON format ONLY.")
        return "\n".join(prompt_parts)

    # --- Domain Classification Method (Example) ---
    async def classify_domain(
        self,
        text: str,
        suggested_domains: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Classify text into hierarchical domain categories using AI."""
        # This still needs a robust implementation if automatic domain classification is desired
        system_prompt = """You are an expert text classifier... Format output ONLY as JSON: { ... }""" # Truncated
        user_prompt = f"Classify the domain hierarchy of the following text:\n\n{text[:5000]}..." # Limit text
        logger.warning("Domain classification called but using placeholder implementation.")
        # Placeholder implementation
        return {"domain_hierarchy": ["General"], "confidence": 0, "reasoning": "Placeholder implementation"}


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
