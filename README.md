# TermExtractor - Advanced Terminology Extraction System

A state-of-the-art terminology extraction system powered by Anthropic's Claude AI, designed for professional translators, terminologists, and localization teams.

## Features

### Core Capabilities
- **AI-Driven Domain Detection**: Hierarchical domain classification with 3+ level depth
- **Custom Domain Targeting**: Manual domain specification and filtering
- **Bilingual & Monolingual Processing**: Extract terms from single or parallel content
- **Multiple Input Methods**: File uploads, direct text, URLs, API submission
- **Large-Scale Processing**: Handle 10+ files with 180,000+ words each
- **Multi-Language Support**: Primary focus on EN↔BG/RO/TR/DE/FR with extensibility
- **Industry-Standard Exports**: TBX, CSV, XLSX, and CAT tool formats

### Advanced Features
- **Hierarchical Domain Classification**: 3+ level deep domain analysis
- **Context-Aware Translations**: Domain-specific translation suggestions
- **Termbase Integration**: Import, verify, and merge with existing termbases
- **Quality Metrics**: Precision, recall, F1 scores, and confidence ratings
- **Configurable Relevance Thresholds**: Filter terms by relevance (0-100%)
- **Comprehensive Metadata**: Statistical, linguistic, semantic, and contextual data

### Performance & Scalability
- **Parallel Processing**: Concurrent file and segment processing
- **Dynamic Batch Optimization**: Configurable batch sizes and priorities
- **Progress Tracking**: Real-time visualization with time estimation
- **Checkpoint/Resume**: Handle interrupted processing
- **Cost Optimization**: Token usage monitoring and caching strategies

### Security & Privacy
- **Data Minimization**: Send only necessary context to API
- **Encrypted Storage**: Secure handling of temporary data
- **GDPR/ISO 27001 Compliance**: Privacy-focused design
- **Secure API Key Management**: Encrypted storage with optional session-only mode
- **Audit Logging**: Complete tracking of data access

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/Termextractor.git
cd Termextractor

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install in development mode
pip install -e .
```

## Quick Start

### Command-Line Interface

```bash
# Extract terms from a file
termextractor extract input.docx --output terms.xlsx

# With custom domain
termextractor extract input.pdf --domain "Medical/Healthcare/Veterinary Medicine" --output terms.tbx

# Bilingual extraction
termextractor extract source.txt --target target.txt --source-lang en --target-lang de --output terms.xlsx

# Batch processing
termextractor batch --input-dir ./files --output-dir ./results --format xlsx
```

### Python API

```python
from termextractor import TermExtractor, DomainClassifier
from termextractor.api import AnthropicClient

# Initialize components
api_client = AnthropicClient(api_key="your-api-key", model="claude-3-5-sonnet-20241022")
extractor = TermExtractor(api_client=api_client)

# Extract terms
results = extractor.extract_from_file(
    file_path="document.docx",
    source_lang="en",
    target_lang="de",
    domain_path="Medical/Healthcare/Veterinary Medicine",
    relevance_threshold=70
)

# Export results
results.export_to_tbx("output.tbx")
results.export_to_excel("output.xlsx")
```

### Web Interface

```bash
# Start the web application
termextractor serve --port 8000

# Access at http://localhost:8000
```

## Configuration

Configuration is managed through `config/config.yaml`:

```yaml
api:
  default_model: "claude-3-5-sonnet-20241022"
  max_tokens_per_request: 4096
  rate_limit_per_minute: 50

processing:
  batch_size: 100
  max_parallel_files: 5
  cache_enabled: true

extraction:
  default_relevance_threshold: 70
  min_term_frequency: 2
  max_term_length: 50

security:
  encrypt_storage: true
  data_retention_days: 7
  audit_logging: true
```

## Supported File Formats

### Translation Formats
- MQXLIFF (.mqxliff)
- SDLXLIFF (.sdlxliff)
- XLIFF (.xliff, .xlf)
- TMX (.tmx)
- TTX (.ttx)

### Document Formats
- Microsoft Word (.docx, .doc)
- PDF (.pdf)
- Rich Text Format (.rtf)
- Plain Text (.txt)
- HTML (.html, .htm)
- XML (.xml)

### Termbase Formats
- TBX (.tbx)
- Excel (.xlsx, .xls)
- CSV (.csv)
- MultiTerm (.sdltb, .mtf)

## Architecture

The system is built with a modular architecture:

- **Core**: AsyncProcessingManager, SecurityManager, ErrorHandlingService, ProgressTracker
- **Extraction**: TermExtractor, DomainClassifier, LanguageProcessor
- **API**: APIManager, AnthropicClient
- **I/O**: FileParser, URLExtractor, FormatExporter
- **Termbase**: TermbaseManager, format handlers
- **Metadata**: MetadataEnrichment
- **Evaluation**: EvaluationTools
- **UI**: CLI, Web interface

See [docs/architecture.md](docs/architecture.md) for detailed documentation.

## Language-Specific Features

- **Germanic Languages**: Compound word analysis
- **Agglutinative Languages**: Morphological analysis
- **Non-Latin Scripts**: Script-specific tokenization (Cyrillic, etc.)
- **RTL Languages**: Hebrew, Arabic support
- **Morphologically Rich**: Slavic, Uralic language handling

## Cost Optimization

- Token usage monitoring and estimation
- Intelligent caching to minimize API calls
- Configurable processing modes (speed vs. quality)
- Cost projection before processing
- Usage analytics dashboard

## API Models

Supported Anthropic models:
- Claude 3.5 Sonnet (Latest) - **Recommended**
- Claude 3.5 Haiku
- Claude 3 Opus
- Claude 3 Haiku
- Claude 3.7 Sonnet

## Contributing

Contributions are welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## License

This project is licensed under the MIT License - see [LICENSE](LICENSE) for details.

## Documentation

- [Architecture Overview](docs/architecture.md)
- [API Reference](docs/api.md)
- [User Guide](docs/user_guide.md)
- [Configuration Guide](docs/configuration.md)
- [Development Guide](docs/development.md)

## Support

For issues and questions:
- GitHub Issues: https://github.com/yourusername/Termextractor/issues
- Documentation: https://termextractor.readthedocs.io

## Acknowledgments

Powered by Anthropic's Claude AI for advanced natural language understanding and terminology extraction.
