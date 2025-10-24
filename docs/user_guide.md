# TermExtractor User Guide

## Table of Contents

1. [Installation](#installation)
2. [Quick Start](#quick-start)
3. [Configuration](#configuration)
4. [Usage Examples](#usage-examples)
5. [Advanced Features](#advanced-features)
6. [Troubleshooting](#troubleshooting)

## Installation

### Requirements

- Python 3.9 or higher
- Anthropic API key
- 500MB free disk space

### Install from source

```bash
# Clone repository
git clone https://github.com/yourusername/Termextractor.git
cd Termextractor

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install package
pip install -e .
```

### Get API Key

1. Sign up at https://console.anthropic.com
2. Create an API key
3. Set environment variable:

```bash
export ANTHROPIC_API_KEY='your-api-key-here'
```

Or create a `.env` file:

```
ANTHROPIC_API_KEY=your-api-key-here
```

## Quick Start

### Extract from a single file

```bash
termextractor extract document.docx --output terms.xlsx --source-lang en
```

### Extract with custom domain

```bash
termextractor extract medical_text.pdf \
  --output medical_terms.xlsx \
  --source-lang en \
  --target-lang de \
  --domain "Medical/Healthcare/Cardiology"
```

### Batch process multiple files

```bash
termextractor batch ./input_files \
  --output-dir ./output \
  --source-lang en \
  --target-lang fr \
  --format xlsx
```

## Configuration

### Configuration File

Create or edit `config/config.yaml`:

```yaml
api:
  default_model: "claude-3-5-sonnet-20241022"
  max_tokens_per_request: 4096
  rate_limit_per_minute: 50

extraction:
  default_relevance_threshold: 70
  min_term_frequency: 2

security:
  encrypt_storage: true
  data_retention_days: 7
```

### Environment Variables

```bash
# Required
ANTHROPIC_API_KEY=your-api-key

# Optional
LOG_LEVEL=INFO
MAX_PARALLEL_FILES=5
BATCH_SIZE=100
```

## Usage Examples

### Example 1: Basic Extraction

Extract English terms from a Word document:

```bash
termextractor extract technical_manual.docx \
  --output terms.xlsx \
  --source-lang en
```

Output: Excel file with all extracted terms

### Example 2: Bilingual Extraction

Extract term pairs (English → German):

```bash
termextractor extract source.txt \
  --output bilingual_terms.tbx \
  --source-lang en \
  --target-lang de
```

Output: TBX file with English terms and German translations

### Example 3: Domain-Specific Extraction

Extract legal terms with custom domain:

```bash
termextractor extract contract.pdf \
  --output legal_terms.csv \
  --source-lang en \
  --domain "Legal/Contract Law/Commercial Contracts" \
  --threshold 80
```

Output: CSV file with high-relevance legal terms only

### Example 4: Batch Processing

Process an entire directory:

```bash
termextractor batch ./documents \
  --output-dir ./results \
  --source-lang en \
  --target-lang bg \
  --format xlsx \
  --threshold 70
```

Output: One Excel file per input document

### Example 5: Python API

```python
import asyncio
from termextractor import TermExtractor
from termextractor.api import AnthropicClient, APIManager

async def main():
    # Initialize
    client = AnthropicClient(
        api_key="your-api-key",
        model="claude-3-5-sonnet-20241022"
    )
    api_manager = APIManager(client=client)
    extractor = TermExtractor(api_client=api_manager)

    # Extract from text
    text = """
    Machine learning is a subset of artificial intelligence that focuses
    on developing algorithms that can learn from data.
    """

    result = await extractor.extract_from_text(
        text=text,
        source_lang="en",
        target_lang="de",
        domain_path="Technology/AI/Machine Learning",
        relevance_threshold=70
    )

    # Print results
    for term in result.terms:
        print(f"{term.term}: {term.translation}")
        print(f"  Relevance: {term.relevance_score}")
        print(f"  Domain: {term.domain}")
        print()

    # Export
    from termextractor.io import FormatExporter
    exporter = FormatExporter()
    await exporter.export(result, "output.xlsx")

asyncio.run(main())
```

## Advanced Features

### Custom Domain Hierarchies

Specify up to 5 levels of domain hierarchy:

```bash
--domain "Medical/Healthcare/Veterinary Medicine/Small Animals/Feline Medicine"
```

### Relevance Threshold Control

Control term filtering with threshold (0-100):

```bash
--threshold 90  # Only very high relevance terms
--threshold 50  # Include more terms
--threshold 0   # All terms
```

### Multiple Output Formats

Export to different formats:

```bash
# Excel with multiple sheets
termextractor extract doc.txt -o terms.xlsx

# TBX for CAT tools
termextractor extract doc.txt -o terms.tbx

# CSV for spreadsheets
termextractor extract doc.txt -o terms.csv

# JSON for APIs
termextractor extract doc.txt -o terms.json
```

### Cost Estimation

Before processing large files, estimate costs:

```python
from termextractor.api import AnthropicClient

client = AnthropicClient(api_key="your-key")

# Read file
with open("large_document.txt", "r") as f:
    text = f.read()

# Estimate cost
estimate = client.estimate_cost(text, num_requests=1)
print(f"Estimated cost: ${estimate['estimated_cost']:.4f}")
print(f"Estimated tokens: {estimate['estimated_total_tokens']:,}")
```

### Progress Tracking

Enable detailed progress visualization:

```python
from termextractor.core import ProgressTracker

tracker = ProgressTracker(
    enable_rich_output=True,
    show_time_estimation=True,
    show_token_usage=True,
    show_cost_estimation=True
)

extractor = TermExtractor(
    api_client=api_manager,
    progress_tracker=tracker
)
```

### Caching

Enable caching to save costs on repeated extractions:

```python
from termextractor.api import APIManager, CacheConfig

cache_config = CacheConfig(
    enabled=True,
    ttl_hours=24,
    max_size_mb=500
)

api_manager = APIManager(
    client=client,
    cache_config=cache_config
)
```

### Rate Limiting

Configure rate limits:

```python
from termextractor.api import RateLimitConfig

rate_config = RateLimitConfig(
    requests_per_minute=50,
    tokens_per_minute=100000,
    max_concurrent_requests=10
)

api_manager = APIManager(
    client=client,
    rate_limit_config=rate_config
)
```

### Security Features

Enable encryption and data minimization:

```python
from termextractor.core import SecurityManager, SecurityConfig

security_config = SecurityConfig(
    encrypt_storage=True,
    data_retention_days=7,
    anonymize_sensitive_data=True,
    audit_logging=True
)

security = SecurityManager(config=security_config)
```

## Troubleshooting

### API Key Not Found

**Error**: `ANTHROPIC_API_KEY not set`

**Solution**:
```bash
export ANTHROPIC_API_KEY='your-api-key'
# Or add to .env file
```

### Rate Limit Exceeded

**Error**: `API rate limit exceeded`

**Solution**:
- Reduce batch size
- Increase delays between requests
- Upgrade API plan

### File Format Not Supported

**Error**: `Unsupported file format: .xyz`

**Solution**:
- Convert to supported format (.txt, .docx, .pdf, .html, .xml)
- Request new format support via GitHub issues

### Out of Memory

**Error**: `MemoryError` during large file processing

**Solution**:
- Process files in smaller batches
- Reduce batch size in config
- Use text chunking

### Low Quality Results

**Issue**: Terms are not relevant

**Solution**:
- Increase relevance threshold
- Specify domain more precisely
- Use higher quality model (Claude 3 Opus)
- Provide better context

### Cost Too High

**Issue**: API costs are excessive

**Solution**:
- Enable caching
- Use Claude 3 Haiku for lower costs
- Reduce batch processing
- Set daily cost limits

## Tips and Best Practices

### 1. Choose the Right Model

- **Claude 3.5 Sonnet**: Best balance (recommended)
- **Claude 3 Opus**: Highest quality, higher cost
- **Claude 3.5 Haiku**: Fastest, lowest cost

### 2. Optimize Domain Specification

Be specific with domains:
- ❌ "Medical"
- ✅ "Medical/Healthcare/Veterinary Medicine"

### 3. Use Appropriate Thresholds

- General use: 70
- High precision needed: 80-90
- Exploratory: 50-60

### 4. Batch Processing

For multiple files:
- Use batch command instead of loops
- Let the system optimize batching
- Enable checkpointing for large jobs

### 5. Cost Management

- Enable caching for repeated extractions
- Use preview/test with small samples first
- Set daily cost limits
- Monitor token usage

### 6. Quality Assurance

- Review high-relevance terms first
- Verify domain classification
- Check translations for accuracy
- Use evaluation metrics

## Getting Help

- Documentation: https://termextractor.readthedocs.io
- Issues: https://github.com/yourusername/Termextractor/issues
- Discussions: https://github.com/yourusername/Termextractor/discussions
