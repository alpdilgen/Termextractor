# TermExtractor Architecture

## Overview

TermExtractor is built with a modular, layered architecture designed for scalability, maintainability, and extensibility.

## Architecture Layers

```
┌─────────────────────────────────────────────────────────┐
│                   User Interfaces                        │
│              (CLI, Web UI, API Endpoints)                │
└─────────────────────────────────────────────────────────┘
                          │
┌─────────────────────────────────────────────────────────┐
│                  Application Layer                       │
│  ┌───────────────┐  ┌──────────────┐  ┌──────────────┐ │
│  │ TermExtractor │  │   Domain     │  │  Language    │ │
│  │               │  │  Classifier  │  │  Processor   │ │
│  └───────────────┘  └──────────────┘  └──────────────┘ │
└─────────────────────────────────────────────────────────┘
                          │
┌─────────────────────────────────────────────────────────┐
│                    API Layer                             │
│  ┌───────────────┐          ┌──────────────────┐        │
│  │  API Manager  │◄────────►│ Anthropic Client │        │
│  │  (Rate Limit, │          │   (Claude API)   │        │
│  │   Caching)    │          └──────────────────┘        │
│  └───────────────┘                                       │
└─────────────────────────────────────────────────────────┘
                          │
┌─────────────────────────────────────────────────────────┐
│                  Core Services                           │
│  ┌──────────────┐ ┌──────────────┐ ┌─────────────────┐ │
│  │   Async      │ │  Security    │ │     Error       │ │
│  │  Processing  │ │   Manager    │ │    Handling     │ │
│  └──────────────┘ └──────────────┘ └─────────────────┘ │
│  ┌──────────────┐ ┌──────────────┐                     │
│  │  Progress    │ │   Metadata   │                     │
│  │   Tracker    │ │  Enrichment  │                     │
│  └──────────────┘ └──────────────┘                     │
└─────────────────────────────────────────────────────────┘
                          │
┌─────────────────────────────────────────────────────────┐
│                   I/O Layer                              │
│  ┌──────────────┐ ┌──────────────┐ ┌─────────────────┐ │
│  │ File Parser  │ │   Format     │ │    Termbase     │ │
│  │ (Multi-fmt)  │ │  Exporter    │ │    Manager      │ │
│  └──────────────┘ └──────────────┘ └─────────────────┘ │
└─────────────────────────────────────────────────────────┘
```

## Component Descriptions

### User Interface Layer

#### CLI (Command-Line Interface)
- **Purpose**: Terminal-based interaction
- **Features**:
  - Single file extraction
  - Batch processing
  - API testing
  - Configuration management
- **Technology**: Click framework

#### Web UI
- **Purpose**: Browser-based interaction
- **Features**:
  - File uploads
  - Visual progress tracking
  - Results visualization
  - Model selection
- **Technology**: Streamlit/FastAPI

#### API Endpoints
- **Purpose**: Programmatic access
- **Features**:
  - REST API
  - JSON responses
  - Authentication
  - Rate limiting

### Application Layer

#### TermExtractor
- **Purpose**: Main terminology extraction engine
- **Responsibilities**:
  - Orchestrate extraction workflow
  - Coordinate with domain classifier and language processor
  - Manage extraction results
  - Filter by relevance thresholds
- **Key Methods**:
  - `extract_from_text()`: Extract from raw text
  - `extract_from_file()`: Extract from files
  - `extract_batch()`: Batch processing
  - `merge_results()`: Merge multiple results

#### DomainClassifier
- **Purpose**: Hierarchical domain detection
- **Responsibilities**:
  - AI-powered domain classification
  - Multi-level hierarchy (up to 5 levels)
  - Confidence scoring
  - Alternative domain suggestions
- **Key Methods**:
  - `classify()`: Classify text into domain
  - `validate_domain_path()`: Validate user input
  - `suggest_subdomains()`: Suggest related domains

#### LanguageProcessor
- **Purpose**: Language-specific processing
- **Responsibilities**:
  - Compound word analysis (Germanic languages)
  - Morphological analysis
  - Script detection
  - Abbreviation extraction
- **Key Methods**:
  - `detect_compounds()`: Find compound words
  - `analyze_morphology()`: Morphological analysis
  - `detect_script()`: Identify writing system
  - `tokenize()`: Language-specific tokenization

### API Layer

#### APIManager
- **Purpose**: High-level API coordination
- **Responsibilities**:
  - Rate limiting (requests and tokens per minute)
  - Response caching
  - Cost tracking and limits
  - Request queuing
- **Key Features**:
  - Smart caching with TTL
  - Automatic retry with backoff
  - Token usage monitoring
  - Daily cost limits

#### AnthropicClient
- **Purpose**: Direct Anthropic API integration
- **Responsibilities**:
  - API communication
  - Token usage tracking
  - Cost calculation
  - Model selection
- **Supported Models**:
  - Claude 3.5 Sonnet (recommended)
  - Claude 3.5 Haiku
  - Claude 3 Opus
  - Claude 3 Haiku
  - Claude 3.7 Sonnet

### Core Services

#### AsyncProcessingManager
- **Purpose**: Concurrent task execution
- **Features**:
  - Parallel file processing
  - Batch optimization
  - Checkpoint/resume
  - Dynamic batch sizing
- **Configuration**:
  - Max parallel workers
  - Batch size (adaptive)
  - Checkpoint intervals

#### SecurityManager
- **Purpose**: Data protection and privacy
- **Features**:
  - AES-256-GCM encryption
  - Secure API key storage
  - Data minimization
  - GDPR compliance
  - Audit logging
- **Compliance**:
  - Data retention policies
  - Right to be forgotten
  - Data export
  - Audit trails

#### ErrorHandlingService
- **Purpose**: Robust error recovery
- **Features**:
  - Automatic error classification
  - Retry logic with exponential backoff
  - Circuit breaker pattern
  - Detailed error logging
- **Error Categories**:
  - API errors
  - Network errors
  - File errors
  - Parsing errors
  - Validation errors

#### ProgressTracker
- **Purpose**: Real-time progress visualization
- **Features**:
  - Rich progress bars
  - Time estimation
  - Token usage tracking
  - Cost estimation
- **Metrics**:
  - Items processed
  - Processing rate
  - Estimated completion time
  - Resource usage

### I/O Layer

#### FileParser
- **Purpose**: Multi-format document parsing
- **Supported Formats**:
  - Plain text (.txt)
  - Microsoft Word (.docx)
  - PDF (.pdf)
  - HTML (.html, .htm)
  - XML (.xml)
- **Future Support**:
  - XLIFF (.xliff)
  - TMX (.tmx)
  - SDLXLIFF (.sdlxliff)

#### FormatExporter
- **Purpose**: Export results to various formats
- **Supported Formats**:
  - Excel (.xlsx) with multiple sheets
  - CSV (.csv)
  - TBX (.tbx) with full metadata
  - JSON (.json)
- **Features**:
  - Dual-output (all terms + high relevance)
  - Metadata inclusion
  - Statistics sheets

#### TermbaseManager
- **Purpose**: Termbase integration
- **Features**:
  - Import from existing termbases
  - Fuzzy matching
  - Cross-verification
  - Conflict resolution

## Data Flow

### Single File Extraction

```
1. User Input
   ↓
2. File Parser → Extract Text
   ↓
3. Domain Classifier → Determine Domain
   ↓
4. Language Processor → Analyze Language Features
   ↓
5. API Manager → Request Term Extraction
   ↓
6. Anthropic API → Generate Results
   ↓
7. Term Parser → Parse and Structure
   ↓
8. Metadata Enrichment → Add Context
   ↓
9. Filter by Relevance → Apply Threshold
   ↓
10. Format Exporter → Export Results
```

### Batch Processing

```
1. Directory Scan → Find Files
   ↓
2. Async Processing Manager → Create Batches
   ↓
3. For Each Batch (Parallel):
   ├─► File 1 → Extract → Cache
   ├─► File 2 → Extract → Cache
   └─► File 3 → Extract → Cache
   ↓
4. Progress Tracker → Update Progress
   ↓
5. Checkpoint → Save State
   ↓
6. Merge Results → Deduplicate
   ↓
7. Export All → Multiple Formats
```

## Design Patterns

### Factory Pattern
- Used in FileParser for format-specific parsers
- Used in FormatExporter for format-specific exporters

### Strategy Pattern
- Used in LanguageProcessor for language-specific algorithms
- Used in ErrorHandlingService for recovery strategies

### Observer Pattern
- Used in ProgressTracker for progress updates
- Used in EventLogger for audit trails

### Singleton Pattern
- Used in SecurityManager for key management
- Used in ConfigurationManager

### Dependency Injection
- All major components accept dependencies via constructor
- Enables testing and modularity

## Scalability Considerations

### Horizontal Scaling
- Stateless design allows multiple instances
- Shared cache via Redis (future)
- Message queue for job distribution (future)

### Vertical Scaling
- Async/await for I/O-bound operations
- Configurable worker pools
- Dynamic batch sizing

### Performance Optimization
- Response caching with TTL
- Batch processing
- Connection pooling
- Lazy loading

## Security Architecture

### Data Protection
```
Input Data
   ↓
Data Minimization → Send only necessary context
   ↓
Encryption (AES-256) → Encrypt sensitive data
   ↓
API Call (TLS) → Secure transmission
   ↓
Response → Parse and store temporarily
   ↓
Auto-Cleanup → Delete after retention period
```

### API Key Management
```
User Input
   ↓
Encryption → Encrypt with master key
   ↓
Secure Storage → OS keyring
   ↓
Runtime Decryption → Decrypt for use
   ↓
Memory Cleanup → Clear after use
```

## Testing Strategy

### Unit Tests
- Test individual components in isolation
- Mock external dependencies
- Test edge cases and error conditions

### Integration Tests
- Test component interactions
- Test API integration
- Test file I/O

### End-to-End Tests
- Test complete workflows
- Test CLI commands
- Test batch processing

## Monitoring and Observability

### Metrics
- API request count
- Token usage
- Cost tracking
- Processing time
- Error rates

### Logging
- Structured logging with loguru
- Multiple log levels (DEBUG, INFO, WARNING, ERROR)
- Log rotation and retention
- Audit logs for security events

### Alerting
- Cost limit alerts
- Rate limit warnings
- Error threshold alerts
- Resource usage warnings
