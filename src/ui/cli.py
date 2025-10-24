"""Command-line interface for TermExtractor."""

import asyncio
import click
from pathlib import Path
from loguru import logger
import os
from typing import Optional

# FIXED: Removed termextractor. prefixes, adjust io->file_io
from api.anthropic_client import AnthropicClient
from api.api_manager import APIManager, RateLimitConfig, CacheConfig # Added RateLimitConfig, CacheConfig
from extraction.term_extractor import TermExtractor
from file_io.file_parser import FileParser # Changed io -> file_io
from file_io.format_exporter import FormatExporter # Changed io -> file_io
from core.progress_tracker import ProgressTracker
from utils.helpers import setup_logging, load_config
from utils.constants import ANTHROPIC_MODELS # Added for model choices


@click.group()
@click.option("--debug", is_flag=True, help="Enable debug logging")
@click.option("--config", type=click.Path(exists=True, path_type=Path), help="Path to config file") # Use Path type
@click.pass_context
def cli(ctx, debug, config):
    """TermExtractor - AI-powered terminology extraction system."""
    log_level = "DEBUG" if debug else "INFO"
    setup_logging(level=log_level)
    cfg = {}
    if config:
        try:
            cfg = load_config(config)
        except Exception as e:
             logger.error(f"Failed to load config file {config}: {e}")
             # Decide whether to exit or continue with defaults
    else:
        # Try default location if exists
        default_config_path = Path.cwd() / "config" / "config.yaml" # Adjust default path as needed
        if default_config_path.exists():
             try:
                  cfg = load_config(default_config_path)
             except Exception as e:
                  logger.warning(f"Failed to load default config {default_config_path}: {e}. Using defaults.")
        else:
             logger.warning("No config file provided or found at default location. Using defaults.")

    # Ensure ctx.obj is a dictionary
    ctx.ensure_object(dict)
    ctx.obj = cfg


@cli.command()
@click.argument("input_file", type=click.Path(exists=True, path_type=Path)) # Use Path type
@click.option("--output", "-o", type=click.Path(path_type=Path), required=True, help="Output file path") # Use Path type
@click.option("--source-lang", "-s", default="en", help="Source language code")
@click.option("--target-lang", "-t", help="Target language code (optional)")
@click.option("--domain", "-d", help="Domain path (e.g., 'Medical/Healthcare')")
@click.option("--threshold", type=float, default=None, help="Relevance threshold (0-100, overrides config/default)")
@click.option("--api-key", envvar="ANTHROPIC_API_KEY", help="Anthropic API key (overrides config)")
@click.option("--model", type=click.Choice(ANTHROPIC_MODELS), default=None, help="Model to use (overrides config)")
@click.pass_context
def extract(ctx, input_file, output, source_lang, target_lang, domain, threshold, api_key, model):
    """Extract terminology from a single file."""
    # Merge config with CLI options (CLI options take precedence)
    config = ctx.obj or {}
    api_key_to_use = api_key or config.get("api", {}).get("anthropic_api_key")
    model_to_use = model or config.get("anthropic", {}).get("default_model", ANTHROPIC_MODELS[0]) # Provide a default
    threshold_to_use = threshold if threshold is not None else config.get("extraction", {}).get("default_relevance_threshold", 70) # Default if not in config

    asyncio.run(_extract(
        input_file, output, source_lang, target_lang, domain, threshold_to_use, api_key_to_use, model_to_use, config
    ))


async def _extract(input_file, output, source_lang, target_lang, domain, threshold, api_key, model, config):
    """Async single file extraction function."""
    if not api_key:
        click.echo("Error: Anthropic API Key not found. Use --api-key, set ANTHROPIC_API_KEY env var, or add to config.", err=True)
        return

    click.echo(f"Starting extraction from: {input_file.name}")
    # ... (print other options) ...

    try:
        # Initialize components using config where available
        rate_limit_cfg = RateLimitConfig(**config.get("api", {}).get("rate_limiting", {}))
        cache_cfg = CacheConfig(**config.get("api", {}).get("caching", {}))

        client = AnthropicClient(api_key=api_key, model=model)
        api_manager = APIManager(client=client, rate_limit_config=rate_limit_cfg, cache_config=cache_cfg)
        tracker = ProgressTracker(enable_rich_output=True) # Enable rich for CLI
        extractor = TermExtractor(
            api_client=api_manager,
            progress_tracker=tracker,
            default_relevance_threshold=threshold, # Pass the final threshold
        )
        exporter = FormatExporter()

        click.echo("Extracting terminology...")
        tracker.start_rich_progress() # Start progress display
        tracker.add_rich_task("extract_file", f"Processing {input_file.name}", total=1)

        result = await extractor.extract_from_file(
            file_path=input_file,
            source_lang=source_lang,
            target_lang=target_lang,
            domain_path=domain,
            relevance_threshold=threshold, # Pass threshold again if method uses it directly
        )

        tracker.update_progress("extract_file", completed=1, message="Exporting...")
        await exporter.export(result, output)
        tracker.complete_task("extract_file")
        tracker.stop_rich_progress() # Stop progress display

        # Print summary
        click.echo("\n" + "="*50)
        click.echo("EXTRACTION SUMMARY")
        click.echo("="*50)
        if "error" in result.metadata:
             click.echo(f"Extraction failed for {input_file.name}: {result.metadata['error']}", err=True)
        else:
             click.echo(f"Total terms extracted (after filter): {len(result.terms)}")
             click.echo(f"High relevance (≥80): {len(result.get_high_relevance_terms())}")
             # ... (print other stats) ...
             click.echo(f"Domain: {' → '.join(result.domain_hierarchy)}")
             usage = client.get_usage_stats()
             click.echo("\nAPI Usage:")
             click.echo(f"  Requests: {usage['total_requests']}")
             click.echo(f"  Tokens: {usage['total_tokens']:,}")
             click.echo(f"  Cost: ${usage['estimated_cost']:.4f}")
             click.echo(f"\nResults saved to: {output}")

    except Exception as e:
        if tracker.progress: tracker.stop_rich_progress() # Ensure progress stops on error
        logger.error(f"Extraction failed: {e}", exc_info=True) # Log traceback
        click.echo(f"\nError during extraction: {e}", err=True)


@cli.command()
@click.argument("input_dir", type=click.Path(exists=True, file_okay=False, path_type=Path)) # Dir exists
@click.option("--output-dir", "-o", type=click.Path(file_okay=False, path_type=Path), required=True, help="Output directory") # Dir path
@click.option("--source-lang", "-s", default="en", help="Source language code")
@click.option("--target-lang", "-t", help="Target language code (optional)")
@click.option("--format", "-f", default="xlsx", help="Output format (xlsx, csv, tbx, json)")
@click.option("--threshold", type=float, default=None, help="Relevance threshold (overrides config/default)")
@click.option("--api-key", envvar="ANTHROPIC_API_KEY", help="Anthropic API key (overrides config)")
@click.option("--model", type=click.Choice(ANTHROPIC_MODELS), default=None, help="Model to use (overrides config)")
@click.option("--max-parallel", type=int, default=None, help="Max parallel files (overrides config)")
@click.pass_context
def batch(ctx, input_dir, output_dir, source_lang, target_lang, format, threshold, api_key, model, max_parallel):
    """Batch process multiple files in a directory."""
    config = ctx.obj or {}
    api_key_to_use = api_key or config.get("api", {}).get("anthropic_api_key")
    model_to_use = model or config.get("anthropic", {}).get("default_model", ANTHROPIC_MODELS[0])
    threshold_to_use = threshold if threshold is not None else config.get("extraction", {}).get("default_relevance_threshold", 70)
    max_parallel_files = max_parallel or config.get("processing", {}).get("max_parallel_files", 5) # Default 5

    asyncio.run(_batch(
        input_dir, output_dir, source_lang, target_lang, format, threshold_to_use, api_key_to_use, model_to_use, max_parallel_files, config
    ))


async def _batch(input_dir, output_dir, source_lang, target_lang, format, threshold, api_key, model, max_parallel_files, config):
    """Async batch processing function."""
    if not api_key:
        click.echo("Error: Anthropic API Key not found.", err=True)
        return

    output_dir.mkdir(parents=True, exist_ok=True)

    # Find supported files (add more extensions as needed)
    supported_extensions = [".txt", ".docx", ".pdf", ".html", ".htm", ".xml", ".xliff", ".sdlxliff", ".mqxliff", ".xlf"]
    files_to_process = [p for p in input_dir.glob("*") if p.is_file() and p.suffix.lower() in supported_extensions]

    if not files_to_process:
         click.echo(f"No supported files found in {input_dir}")
         return

    click.echo(f"Found {len(files_to_process)} files to process with max {max_parallel_files} parallel workers.")

    # Initialize components
    rate_limit_cfg = RateLimitConfig(**config.get("api", {}).get("rate_limiting", {}))
    cache_cfg = CacheConfig(**config.get("api", {}).get("caching", {}))
    client = AnthropicClient(api_key=api_key, model=model)
    # Note: Using one APIManager means rate limits/costs are shared across parallel tasks
    api_manager = APIManager(client=client, rate_limit_config=rate_limit_cfg, cache_config=cache_cfg)
    tracker = ProgressTracker(enable_rich_output=True)
    extractor = TermExtractor(
        api_client=api_manager, progress_tracker=tracker, # Pass tracker to extractor
        default_relevance_threshold=threshold,
    )
    exporter = FormatExporter()

    tracker.start_rich_progress()
    main_task_id = tracker.add_rich_task("batch_process", "Processing files", total=len(files_to_process))

    # Define the function to process a single file
    async def process_single_file(file_path: Path):
        file_task_id = f"file_{file_path.stem}"
        # No need to add individual rich tasks if using a single overall progress bar
        # tracker.add_rich_task(file_task_id, f"Processing {file_path.name}", total=1)
        try:
            result = await extractor.extract_from_file(
                file_path=file_path,
                source_lang=source_lang,
                target_lang=target_lang,
                relevance_threshold=threshold,
                # Domain is not passed in batch, could be added as option
            )
            output_file = output_dir / f"{file_path.stem}_terms.{format}"
            await exporter.export(result, output_file)
            # tracker.update_progress(file_task_id, completed=1)
            # tracker.complete_task(file_task_id)
            if "error" in result.metadata:
                 logger.error(f"Failed processing {file_path.name}: {result.metadata['error']}")
                 return False # Indicate failure
            else:
                 logger.info(f"Successfully processed {file_path.name}, found {len(result.terms)} terms.")
                 return True # Indicate success
        except Exception as e:
            # tracker.complete_task(file_task_id, success=False)
            logger.error(f"Unhandled error processing {file_path.name}: {e}", exc_info=True)
            return False # Indicate failure
        finally:
            # Update main progress bar regardless of success/failure
             tracker.update_progress(main_task_id, advance=1)


    # Run tasks with concurrency limit
    semaphore = asyncio.Semaphore(max_parallel_files)
    async def run_with_semaphore(file_path):
         async with semaphore:
              return await process_single_file(file_path)

    tasks = [run_with_semaphore(fp) for fp in files_to_process]
    results = await asyncio.gather(*tasks)

    tracker.complete_task(main_task_id)
    tracker.stop_rich_progress()

    # Final summary
    success_count = sum(1 for r in results if r is True)
    fail_count = len(results) - success_count
    usage = client.get_usage_stats()
    click.echo("\n" + "="*50)
    click.echo("BATCH PROCESSING SUMMARY")
    click.echo("="*50)
    click.echo(f"Total files: {len(files_to_process)}")
    click.echo(f"Successfully processed: {success_count}")
    click.echo(f"Failed: {fail_count}", err=(fail_count > 0))
    click.echo(f"\nTotal API Usage:")
    click.echo(f"  Requests: {usage['total_requests']}")
    click.echo(f"  Tokens: {usage['total_tokens']:,}")
    click.echo(f"  Cost: ${usage['estimated_cost']:.4f}")
    click.echo(f"\nResults saved to directory: {output_dir}")

# ... (keep test_api command and main function) ...
@cli.command()
@click.option("--api-key", envvar="ANTHROPIC_API_KEY", help="Anthropic API key")
@click.option("--model", type=click.Choice(ANTHROPIC_MODELS), default=ANTHROPIC_MODELS[0], help="Model to test")
def test_api(api_key, model):
     """Test API connection."""
     # ... (implementation seems okay) ...
     pass

def main():
     """Main entry point."""
     cli(obj={})

if __name__ == "__main__":
    main()
