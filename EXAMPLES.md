# TermExtractor Examples

This document provides comprehensive examples of using TermExtractor for various use cases.

## Table of Contents

1. [Basic Examples](#basic-examples)
2. [Advanced Examples](#advanced-examples)
3. [Industry-Specific Examples](#industry-specific-examples)
4. [Python API Examples](#python-api-examples)

## Basic Examples

### Example 1: Extract Terms from English Text

```bash
termextractor extract document.txt \
  --output terms.xlsx \
  --source-lang en
```

**Input** (document.txt):
```
Cloud computing is the delivery of computing services over the internet.
It includes servers, storage, databases, networking, software, and analytics.
```

**Output** (terms.xlsx):
| Term | Domain | Relevance | POS |
|------|--------|-----------|-----|
| cloud computing | Technology/IT | 95 | NOUN |
| computing services | Technology/IT | 88 | NOUN |
| internet | Technology/Networks | 82 | NOUN |
| databases | Technology/IT/Data | 90 | NOUN |

### Example 2: Bilingual Extraction (EN → DE)

```bash
termextractor extract technical_doc.pdf \
  --output bilingual_terms.tbx \
  --source-lang en \
  --target-lang de
```

**Input**: Technical documentation in English

**Output** (bilingual_terms.tbx):
```xml
<termEntry>
  <langSet xml:lang="en">
    <term>database management system</term>
  </langSet>
  <langSet xml:lang="de">
    <term>Datenbankverwaltungssystem</term>
  </langSet>
</termEntry>
```

### Example 3: Extract with Custom Domain

```bash
termextractor extract medical_report.docx \
  --output medical_terms.csv \
  --source-lang en \
  --domain "Medical/Healthcare/Cardiology" \
  --threshold 80
```

**Output**: Only cardiology-specific terms with 80+ relevance score

## Advanced Examples

### Example 4: Batch Processing with Progress Tracking

```bash
termextractor batch ./medical_documents \
  --output-dir ./extracted_terms \
  --source-lang en \
  --target-lang ro \
  --format xlsx \
  --threshold 75
```

**Directory Structure**:
```
medical_documents/
  ├── patient_report_1.pdf
  ├── patient_report_2.pdf
  └── guidelines.docx

extracted_terms/
  ├── patient_report_1_terms.xlsx
  ├── patient_report_2_terms.xlsx
  └── guidelines_terms.xlsx
```

### Example 5: Multi-Level Domain Hierarchy

```bash
termextractor extract contract.pdf \
  --output contract_terms.xlsx \
  --source-lang en \
  --domain "Legal/Contract Law/Commercial Contracts/Intellectual Property" \
  --threshold 85
```

**Result**: Terms specific to IP clauses in commercial contracts

## Industry-Specific Examples

### Medical/Healthcare

#### Veterinary Medicine Example

```bash
termextractor extract vet_manual.pdf \
  --output vet_terms.xlsx \
  --source-lang en \
  --target-lang bg \
  --domain "Medical/Healthcare/Veterinary Medicine/Animal Pharmacology"
```

**Sample Output**:
```
antiparasitic drug → антипаразитно лекарство
dosage regimen → дозов режим
therapeutic efficacy → терапевтична ефективност
adverse reaction → нежелана реакция
```

### Legal

#### Contract Translation Example

```bash
termextractor extract nda.docx \
  --output nda_terms.tbx \
  --source-lang en \
  --target-lang de \
  --domain "Legal/Contract Law/Confidentiality Agreements"
```

**Sample Output**:
```
confidential information → vertrauliche Informationen
non-disclosure obligation → Geheimhaltungspflicht
proprietary data → geschützte Daten
```

### Technology

#### Software Documentation Example

```bash
termextractor extract api_docs.html \
  --output api_terms.json \
  --source-lang en \
  --domain "Technology/Software Development/API Design"
```

**Sample Output**:
```json
{
  "terms": [
    {
      "term": "RESTful API",
      "domain": "Technology/Software Development/API Design",
      "relevance_score": 98,
      "definition": "API that follows REST architectural style"
    },
    {
      "term": "endpoint",
      "domain": "Technology/Software Development/API Design",
      "relevance_score": 95,
      "definition": "Specific URL where API can be accessed"
    }
  ]
}
```

### Finance

#### Banking Terms Example

```bash
termextractor extract financial_report.pdf \
  --output banking_terms.xlsx \
  --source-lang en \
  --target-lang fr \
  --domain "Finance/Banking/Retail Banking"
```

**Sample Output**:
```
credit facility → facilité de crédit
loan-to-value ratio → ratio prêt/valeur
collateral → garantie
interest rate → taux d'intérêt
```

## Python API Examples

### Example 6: Simple Python Script

```python
import asyncio
from termextractor import TermExtractor
from termextractor.api import AnthropicClient, APIManager

async def extract_terms():
    # Setup
    client = AnthropicClient(api_key="your-api-key")
    api_manager = APIManager(client=client)
    extractor = TermExtractor(api_client=api_manager)

    # Extract
    result = await extractor.extract_from_text(
        text="Kubernetes is a container orchestration platform.",
        source_lang="en",
        relevance_threshold=70
    )

    # Print
    for term in result.terms:
        print(f"{term.term} (Score: {term.relevance_score})")

asyncio.run(extract_terms())
```

### Example 7: Custom Configuration

```python
import asyncio
from pathlib import Path
from termextractor import TermExtractor
from termextractor.api import (
    AnthropicClient,
    APIManager,
    RateLimitConfig,
    CacheConfig
)
from termextractor.core import ProgressTracker

async def main():
    # Configure API
    client = AnthropicClient(
        api_key="your-api-key",
        model="claude-3-5-sonnet-20241022",
        max_tokens=4096,
        temperature=0.0
    )

    # Configure rate limiting
    rate_config = RateLimitConfig(
        requests_per_minute=40,
        tokens_per_minute=80000,
        max_concurrent_requests=5
    )

    # Configure caching
    cache_config = CacheConfig(
        enabled=True,
        ttl_hours=48,
        max_size_mb=1000
    )

    # Create API manager
    api_manager = APIManager(
        client=client,
        rate_limit_config=rate_config,
        cache_config=cache_config,
        max_cost_per_day=10.0  # $10 daily limit
    )

    # Create progress tracker
    tracker = ProgressTracker(
        enable_rich_output=True,
        show_time_estimation=True,
        show_token_usage=True
    )

    # Create extractor
    extractor = TermExtractor(
        api_client=api_manager,
        progress_tracker=tracker,
        default_relevance_threshold=75
    )

    # Extract from file
    result = await extractor.extract_from_file(
        file_path=Path("document.pdf"),
        source_lang="en",
        target_lang="tr",
        domain_path="Technology/Software/Cloud Computing"
    )

    # Export results
    from termextractor.io import FormatExporter
    exporter = FormatExporter()
    await exporter.export(result, Path("output.xlsx"))

    # Print statistics
    print(f"\nExtraction Statistics:")
    print(f"Total terms: {len(result.terms)}")
    print(f"High relevance: {len(result.get_high_relevance_terms())}")
    print(f"Domain: {' → '.join(result.domain_hierarchy)}")

    # Print API usage
    usage = client.get_usage_stats()
    print(f"\nAPI Usage:")
    print(f"Requests: {usage['total_requests']}")
    print(f"Tokens: {usage['total_tokens']:,}")
    print(f"Cost: ${usage['estimated_cost']:.4f}")

asyncio.run(main())
```

### Example 8: Batch Processing with Progress

```python
import asyncio
from pathlib import Path
from termextractor import TermExtractor
from termextractor.api import AnthropicClient, APIManager
from termextractor.core import ProgressTracker

async def batch_extract():
    # Setup
    client = AnthropicClient(api_key="your-api-key")
    api_manager = APIManager(client=client)
    tracker = ProgressTracker(enable_rich_output=True)
    extractor = TermExtractor(
        api_client=api_manager,
        progress_tracker=tracker
    )

    # Get all files
    input_dir = Path("./documents")
    files = list(input_dir.glob("*.pdf"))

    # Start progress tracking
    tracker.start_rich_progress()

    # Process each file
    for file_path in files:
        print(f"\nProcessing: {file_path.name}")

        result = await extractor.extract_from_file(
            file_path=file_path,
            source_lang="en",
            target_lang="de"
        )

        # Export
        output_path = Path(f"./output/{file_path.stem}_terms.xlsx")
        output_path.parent.mkdir(exist_ok=True)

        from termextractor.io import FormatExporter
        exporter = FormatExporter()
        await exporter.export(result, output_path)

        print(f"Extracted {len(result.terms)} terms")

    tracker.stop_rich_progress()

    # Final statistics
    usage = client.get_usage_stats()
    print(f"\n{'='*50}")
    print(f"Batch Processing Complete")
    print(f"{'='*50}")
    print(f"Files processed: {len(files)}")
    print(f"Total cost: ${usage['estimated_cost']:.4f}")

asyncio.run(batch_extract())
```

### Example 9: Domain Classification

```python
import asyncio
from termextractor.extraction import DomainClassifier
from termextractor.api import AnthropicClient, APIManager

async def classify_domains():
    client = AnthropicClient(api_key="your-api-key")
    api_manager = APIManager(client=client)
    classifier = DomainClassifier(api_client=api_manager)

    texts = [
        "The patient presents with acute myocardial infarction.",
        "The API endpoint returns a JSON response with status code 200.",
        "The defendant violated the non-compete clause."
    ]

    for text in texts:
        result = await classifier.classify(text)
        print(f"\nText: {text}")
        print(f"Domain: {result.full_path}")
        print(f"Confidence: {result.confidence:.2f}")
        print(f"Keywords: {', '.join(result.keywords)}")

asyncio.run(classify_domains())
```

### Example 10: Cost Estimation Before Processing

```python
import asyncio
from pathlib import Path
from termextractor.api import AnthropicClient

async def estimate_cost():
    client = AnthropicClient(api_key="your-api-key")

    # Read file
    file_path = Path("large_document.txt")
    with open(file_path, "r") as f:
        text = f.read()

    # Estimate cost
    estimate = client.estimate_cost(text, num_requests=1)

    print("Cost Estimation:")
    print(f"  Input tokens: {estimate['estimated_input_tokens']:,}")
    print(f"  Output tokens: {estimate['estimated_output_tokens']:,}")
    print(f"  Total tokens: {estimate['estimated_total_tokens']:,}")
    print(f"  Estimated cost: ${estimate['estimated_cost']:.4f}")

    # Ask for confirmation
    response = input("\nProceed with extraction? (y/n): ")
    if response.lower() == 'y':
        # Proceed with extraction
        print("Starting extraction...")
    else:
        print("Extraction cancelled.")

asyncio.run(estimate_cost())
```

## Testing Examples

### Example 11: Test API Connection

```bash
termextractor test-api --model claude-3-5-sonnet-20241022
```

**Output**:
```
Testing API connection with model: claude-3-5-sonnet-20241022
✓ API connection successful!
Response: OK
Tokens used: 15
Cost: $0.000068
```

### Example 12: Test with Sample Text

```python
import asyncio
from termextractor import TermExtractor
from termextractor.api import AnthropicClient, APIManager

async def test_extraction():
    client = AnthropicClient(api_key="your-api-key")
    api_manager = APIManager(client=client)
    extractor = TermExtractor(api_client=api_manager)

    # Small test text
    test_text = "Machine learning algorithms process large datasets."

    result = await extractor.extract_from_text(
        text=test_text,
        source_lang="en",
        relevance_threshold=50  # Lower threshold for testing
    )

    print(f"Test successful! Found {len(result.terms)} terms")
    for term in result.terms:
        print(f"  - {term.term} ({term.relevance_score})")

asyncio.run(test_extraction())
```

## Troubleshooting Examples

### Example 13: Handle Rate Limiting

```python
import asyncio
from termextractor.api import (
    AnthropicClient,
    APIManager,
    RateLimitConfig
)
from termextractor.core import ErrorHandlingService

async def extract_with_retry():
    client = AnthropicClient(api_key="your-api-key")

    # Conservative rate limits
    rate_config = RateLimitConfig(
        requests_per_minute=30,  # Lower than default
        tokens_per_minute=50000
    )

    api_manager = APIManager(
        client=client,
        rate_limit_config=rate_config
    )

    # Error handling
    error_handler = ErrorHandlingService(
        max_retries=5,  # More retries
        retry_backoff_factor=3.0  # Longer waits
    )

    # Use error handler for retry logic
    result = await error_handler.retry_with_backoff(
        api_manager.extract_terms,
        text="Your text here",
        source_lang="en"
    )

    print("Extraction completed successfully")

asyncio.run(extract_with_retry())
```

For more examples and updates, visit:
https://github.com/yourusername/Termextractor/tree/main/examples
