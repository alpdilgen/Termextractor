"""Command-line interface for TermExtractor."""

import asyncio
import click
from pathlib import Path
from loguru import logger
import os
from typing import Optional

from termextractor.api.anthropic_client import AnthropicClient
from termextractor.api.api_manager import APIManager
from termextractor.extraction.term_extractor import TermExtractor
from termextractor.io.file_parser import FileParser
from termextractor.io.format_exporter import FormatExporter
from termextractor.core.progress_tracker import ProgressTracker
from termextractor.utils.helpers import setup_logging, load_config


@click.group()
@click.option("--debug", is_flag=True, help="Enable debug logging")
@click.option("--config", type=click.Path(exists=True), help="Path to config file")
@click.pass_context
def cli(ctx, debug, config):
    """TermExtractor - AI-powered terminology extraction system."""
    # Setup logging
    log_level = "DEBUG" if debug else "INFO"
    setup_logging(level=log_level)

    # Load config
    if config:
        ctx.obj = load_config(config)
    else:
        try:
            ctx.obj = load_config()
        except FileNotFoundError:
            ctx.obj = {}
            logger.warning("No config file found, using defaults")


@cli.command()
@click.argument("input_file", type=click.Path(exists=True))
@click.option("--output", "-o", type=click.Path(), required=True, help="Output file path")
@click.option("--source-lang", "-s", default="en", help="Source language code")
@click.option("--target-lang", "-t", help="Target language code (optional)")
@click.option("--domain", "-d", help="Domain path (e.g., 'Medical/Healthcare')")
@click.option("--threshold", type=float, default=70, help="Relevance threshold (0-100)")
@click.option("--api-key", envvar="ANTHROPIC_API_KEY", help="Anthropic API key")
@click.option("--model", default="claude-3-5-sonnet-20241022", help="Model to use")
@click.pass_context
def extract(ctx, input_file, output, source_lang, target_lang, domain, threshold, api_key, model):
    """Extract terminology from a file."""
    asyncio.run(_extract(
        ctx, input_file, output, source_lang, target_lang, domain, threshold, api_key, model
    ))


async def _extract(ctx, input_file, output, source_lang, target_lang, domain, threshold, api_key, model):
    """Async extraction function."""
    if not api_key:
        click.echo("Error: ANTHROPIC_API_KEY not set. Use --api-key or set environment variable.", err=True)
        return

    click.echo(f"Extracting terms from: {input_file}")
    click.echo(f"Source language: {source_lang}")
    if target_lang:
        click.echo(f"Target language: {target_lang}")
    if domain:
        click.echo(f"Domain: {domain}")

    # Initialize components
    client = AnthropicClient(api_key=api_key, model=model)
    api_manager = APIManager(client=client)
    tracker = ProgressTracker(enable_rich_output=True)
    extractor = TermExtractor(
        api_client=api_manager,
        progress_tracker=tracker,
        default_relevance_threshold=threshold,
    )

    try:
        # Extract terms
        click.echo("\nExtracting terminology...")
        result = await extractor.extract_from_file(
            file_path=Path(input_file),
            source_lang=source_lang,
            target_lang=target_lang,
            domain_path=domain,
            relevance_threshold=threshold,
        )

        # Export results
        click.echo(f"\nExporting to: {output}")
        exporter = FormatExporter()
        await exporter.export(result, Path(output))

        # Print summary
        click.echo("\n" + "="*50)
        click.echo("EXTRACTION SUMMARY")
        click.echo("="*50)
        click.echo(f"Total terms extracted: {len(result.terms)}")
        click.echo(f"High relevance (≥80): {len(result.get_high_relevance_terms())}")
        click.echo(f"Medium relevance (60-79): {len(result.get_medium_relevance_terms())}")
        click.echo(f"Low relevance (<60): {len(result.get_low_relevance_terms())}")
        click.echo(f"Domain: {' → '.join(result.domain_hierarchy)}")

        # API usage
        usage = client.get_usage_stats()
        click.echo(f"\nAPI Usage:")
        click.echo(f"  Requests: {usage['total_requests']}")
        click.echo(f"  Tokens: {usage['total_tokens']:,}")
        click.echo(f"  Cost: ${usage['estimated_cost']:.4f}")

        click.echo(f"\nResults saved to: {output}")

    except Exception as e:
        logger.error(f"Extraction failed: {e}")
        click.echo(f"Error: {e}", err=True)
        raise


@cli.command()
@click.argument("input_dir", type=click.Path(exists=True))
@click.option("--output-dir", "-o", type=click.Path(), required=True, help="Output directory")
@click.option("--source-lang", "-s", default="en", help="Source language code")
@click.option("--target-lang", "-t", help="Target language code (optional)")
@click.option("--format", "-f", default="xlsx", help="Output format (xlsx, csv, tbx, json)")
@click.option("--threshold", type=float, default=70, help="Relevance threshold")
@click.option("--api-key", envvar="ANTHROPIC_API_KEY", help="Anthropic API key")
@click.option("--model", default="claude-3-5-sonnet-20241022", help="Model to use")
@click.pass_context
def batch(ctx, input_dir, output_dir, source_lang, target_lang, format, threshold, api_key, model):
    """Batch process multiple files."""
    asyncio.run(_batch(
        ctx, input_dir, output_dir, source_lang, target_lang, format, threshold, api_key, model
    ))


async def _batch(ctx, input_dir, output_dir, source_lang, target_lang, format, threshold, api_key, model):
    """Async batch processing function."""
    if not api_key:
        click.echo("Error: ANTHROPIC_API_KEY not set.", err=True)
        return

    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Find all files
    files = list(input_path.glob("*.*"))
    click.echo(f"Found {len(files)} files to process")

    # Initialize components
    client = AnthropicClient(api_key=api_key, model=model)
    api_manager = APIManager(client=client)
    tracker = ProgressTracker(enable_rich_output=True)
    extractor = TermExtractor(
        api_client=api_manager,
        progress_tracker=tracker,
        default_relevance_threshold=threshold,
    )
    exporter = FormatExporter()

    # Process each file
    for idx, file_path in enumerate(files, 1):
        click.echo(f"\n[{idx}/{len(files)}] Processing: {file_path.name}")

        try:
            result = await extractor.extract_from_file(
                file_path=file_path,
                source_lang=source_lang,
                target_lang=target_lang,
                relevance_threshold=threshold,
            )

            # Export
            output_file = output_path / f"{file_path.stem}_terms.{format}"
            await exporter.export(result, output_file)

            click.echo(f"  ✓ Extracted {len(result.terms)} terms")

        except Exception as e:
            logger.error(f"Failed to process {file_path}: {e}")
            click.echo(f"  ✗ Error: {e}", err=True)

    # Final summary
    usage = client.get_usage_stats()
    click.echo("\n" + "="*50)
    click.echo(f"Processed {len(files)} files")
    click.echo(f"Total API cost: ${usage['estimated_cost']:.4f}")
    click.echo(f"Results saved to: {output_dir}")


@cli.command()
@click.option("--api-key", envvar="ANTHROPIC_API_KEY", help="Anthropic API key")
@click.option("--model", default="claude-3-5-sonnet-20241022", help="Model to test")
def test_api(api_key, model):
    """Test API connection."""
    asyncio.run(_test_api(api_key, model))


async def _test_api(api_key, model):
    """Async API test function."""
    if not api_key:
        click.echo("Error: ANTHROPIC_API_KEY not set.", err=True)
        return

    click.echo(f"Testing API connection with model: {model}")

    try:
        client = AnthropicClient(api_key=api_key, model=model)
        response = await client.generate_text(
            prompt="Hello, please respond with 'OK' to confirm connection.",
            max_tokens=10,
        )

        click.echo("✓ API connection successful!")
        click.echo(f"Response: {response.content}")
        click.echo(f"Tokens used: {response.input_tokens + response.output_tokens}")
        click.echo(f"Cost: ${response.cost:.6f}")

    except Exception as e:
        click.echo(f"✗ API connection failed: {e}", err=True)


def main():
    """Main entry point."""
    cli(obj={})


if __name__ == "__main__":
    main()
