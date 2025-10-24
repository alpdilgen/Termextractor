"""Streamlit web application for TermExtractor."""

import streamlit as st
import asyncio
import tempfile
from pathlib import Path
import pandas as pd
from datetime import datetime
import os

from termextractor.api.anthropic_client import AnthropicClient
from termextractor.api.api_manager import APIManager, RateLimitConfig, CacheConfig
from termextractor.extraction.term_extractor import TermExtractor
from termextractor.core.progress_tracker import ProgressTracker
from termextractor.io.format_exporter import FormatExporter
from termextractor.utils.constants import ANTHROPIC_MODELS, SUPPORTED_LANGUAGES
from termextractor.utils.helpers import load_config

# Page configuration
st.set_page_config(
    page_title="TermExtractor - AI-Powered Terminology Extraction",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .success-box {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        border-radius: 0.25rem;
        padding: 1rem;
        margin: 1rem 0;
    }
    .warning-box {
        background-color: #fff3cd;
        border: 1px solid #ffeaa7;
        border-radius: 0.25rem;
        padding: 1rem;
        margin: 1rem 0;
    }
    .error-box {
        background-color: #f8d7da;
        border: 1px solid #f5c6cb;
        border-radius: 0.25rem;
        padding: 1rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)


def init_session_state():
    """Initialize session state variables."""
    if 'extraction_results' not in st.session_state:
        st.session_state.extraction_results = None
    if 'api_usage' not in st.session_state:
        st.session_state.api_usage = None
    if 'processing' not in st.session_state:
        st.session_state.processing = False


def get_api_key():
    """Get API key from session state or environment."""
    if 'api_key' in st.session_state and st.session_state.api_key:
        return st.session_state.api_key
    return os.getenv('ANTHROPIC_API_KEY', '')


async def extract_terms_async(
    file_content,
    file_name,
    source_lang,
    target_lang,
    domain_path,
    threshold,
    model,
    api_key
):
    """Async function to extract terms."""
    # Save uploaded file temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=Path(file_name).suffix) as tmp_file:
        tmp_file.write(file_content)
        tmp_path = Path(tmp_file.name)

    try:
        # Initialize components
        client = AnthropicClient(api_key=api_key, model=model)

        # Configure API manager
        rate_config = RateLimitConfig(
            requests_per_minute=50,
            tokens_per_minute=100000,
            max_concurrent_requests=10
        )

        cache_config = CacheConfig(
            enabled=True,
            ttl_hours=24,
            max_size_mb=500
        )

        api_manager = APIManager(
            client=client,
            rate_limit_config=rate_config,
            cache_config=cache_config
        )

        tracker = ProgressTracker(enable_rich_output=False)

        extractor = TermExtractor(
            api_client=api_manager,
            progress_tracker=tracker,
            default_relevance_threshold=threshold
        )

        # Extract terms
        result = await extractor.extract_from_file(
            file_path=tmp_path,
            source_lang=source_lang,
            target_lang=target_lang if target_lang else None,
            domain_path=domain_path if domain_path else None,
            relevance_threshold=threshold
        )

        # Get usage stats
        usage = client.get_usage_stats()

        return result, usage

    finally:
        # Clean up temp file
        if tmp_path.exists():
            tmp_path.unlink()


def display_results(result, usage):
    """Display extraction results."""
    st.markdown("---")
    st.markdown("## 📊 Extraction Results")

    # Statistics
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Total Terms", len(result.terms))

    with col2:
        high_relevance = len(result.get_high_relevance_terms())
        st.metric("High Relevance (≥80)", high_relevance)

    with col3:
        medium_relevance = len(result.get_medium_relevance_terms())
        st.metric("Medium Relevance (60-79)", medium_relevance)

    with col4:
        low_relevance = len(result.get_low_relevance_terms())
        st.metric("Low Relevance (<60)", low_relevance)

    # Domain information
    st.markdown("### 🎯 Domain Classification")
    domain_path = " → ".join(result.domain_hierarchy)
    st.info(f"**Domain:** {domain_path}")

    # API Usage
    st.markdown("### 💰 API Usage")
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Total Requests", usage['total_requests'])

    with col2:
        st.metric("Total Tokens", f"{usage['total_tokens']:,}")

    with col3:
        st.metric("Estimated Cost", f"${usage['estimated_cost']:.4f}")

    # Terms table
    st.markdown("### 📝 Extracted Terms")

    # Filter options
    filter_option = st.selectbox(
        "Filter by relevance:",
        ["All Terms", "High Relevance (≥80)", "Medium Relevance (60-79)", "Low Relevance (<60)"]
    )

    if filter_option == "High Relevance (≥80)":
        filtered_terms = result.get_high_relevance_terms()
    elif filter_option == "Medium Relevance (60-79)":
        filtered_terms = result.get_medium_relevance_terms()
    elif filter_option == "Low Relevance (<60)":
        filtered_terms = result.get_low_relevance_terms()
    else:
        filtered_terms = result.terms

    if filtered_terms:
        # Convert to DataFrame
        df_data = []
        for term in filtered_terms:
            df_data.append({
                "Term": term.term,
                "Translation": term.translation or "",
                "Domain": term.domain,
                "Subdomain": term.subdomain or "",
                "Part of Speech": term.pos,
                "Relevance Score": term.relevance_score,
                "Confidence Score": term.confidence_score,
                "Frequency": term.frequency,
                "Definition": term.definition,
                "Context": term.context
            })

        df = pd.DataFrame(df_data)

        # Display with pagination
        st.dataframe(
            df,
            use_container_width=True,
            height=400
        )

        # Download options
        st.markdown("### 💾 Download Results")

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            # Excel export
            if st.button("📊 Download as Excel", use_container_width=True):
                with st.spinner("Generating Excel file..."):
                    export_file = tempfile.NamedTemporaryFile(delete=False, suffix=".xlsx")
                    exporter = FormatExporter()
                    asyncio.run(exporter.export(result, Path(export_file.name)))

                    with open(export_file.name, "rb") as f:
                        st.download_button(
                            label="⬇️ Download XLSX",
                            data=f.read(),
                            file_name=f"terms_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                        )

        with col2:
            # CSV export
            csv = df.to_csv(index=False)
            st.download_button(
                label="📄 Download as CSV",
                data=csv,
                file_name=f"terms_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
                use_container_width=True
            )

        with col3:
            # JSON export
            if st.button("📋 Download as JSON", use_container_width=True):
                with st.spinner("Generating JSON file..."):
                    import json
                    json_data = json.dumps(result.to_dict(), indent=2)
                    st.download_button(
                        label="⬇️ Download JSON",
                        data=json_data,
                        file_name=f"terms_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                        mime="application/json"
                    )

        with col4:
            # TBX export
            if st.button("🗂️ Download as TBX", use_container_width=True):
                with st.spinner("Generating TBX file..."):
                    export_file = tempfile.NamedTemporaryFile(delete=False, suffix=".tbx")
                    exporter = FormatExporter()
                    asyncio.run(exporter.export(result, Path(export_file.name)))

                    with open(export_file.name, "rb") as f:
                        st.download_button(
                            label="⬇️ Download TBX",
                            data=f.read(),
                            file_name=f"terms_{datetime.now().strftime('%Y%m%d_%H%M%S')}.tbx",
                            mime="application/xml"
                        )
    else:
        st.warning("No terms match the selected filter.")


def main():
    """Main Streamlit application."""
    init_session_state()

    # Header
    st.markdown('<div class="main-header">📚 TermExtractor</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-header">AI-Powered Terminology Extraction with Claude</div>',
        unsafe_allow_html=True
    )

    # Sidebar
    with st.sidebar:
        st.markdown("## ⚙️ Configuration")

        # API Key
        st.markdown("### 🔑 API Key")
        api_key_input = st.text_input(
            "Anthropic API Key",
            type="password",
            value=get_api_key(),
            help="Enter your Anthropic API key. It will not be stored."
        )

        if api_key_input:
            st.session_state.api_key = api_key_input
            st.success("✓ API Key set")
        else:
            st.warning("⚠️ API Key required")

        st.markdown("---")

        # Model Selection
        st.markdown("### 🤖 Model")
        model = st.selectbox(
            "Select Claude Model",
            ANTHROPIC_MODELS,
            index=0,
            help="Claude 3.5 Sonnet is recommended for best quality"
        )

        # Cost info
        model_costs = {
            "claude-3-5-sonnet-20241022": "$$$ (Best quality)",
            "claude-3-5-haiku-20241022": "$ (Fastest)",
            "claude-3-opus-20240229": "$$$$ (Premium)",
            "claude-3-haiku-20240307": "$ (Economy)",
        }
        if model in model_costs:
            st.info(f"💰 {model_costs[model]}")

        st.markdown("---")

        # Language Settings
        st.markdown("### 🌍 Languages")

        source_lang = st.selectbox(
            "Source Language",
            list(SUPPORTED_LANGUAGES.keys()),
            format_func=lambda x: f"{x.upper()} - {SUPPORTED_LANGUAGES[x]}"
        )

        enable_bilingual = st.checkbox("Bilingual Extraction", value=False)

        target_lang = None
        if enable_bilingual:
            target_lang = st.selectbox(
                "Target Language",
                list(SUPPORTED_LANGUAGES.keys()),
                format_func=lambda x: f"{x.upper()} - {SUPPORTED_LANGUAGES[x]}",
                index=1
            )

        st.markdown("---")

        # Domain Settings
        st.markdown("### 🎯 Domain")

        enable_custom_domain = st.checkbox("Custom Domain Path", value=False)

        domain_path = None
        if enable_custom_domain:
            domain_path = st.text_input(
                "Domain Hierarchy",
                placeholder="e.g., Medical/Healthcare/Cardiology",
                help="Use / to separate levels. Example: Technology/AI/Machine Learning"
            )

            st.caption("Examples:")
            st.caption("• Medical/Healthcare/Veterinary Medicine")
            st.caption("• Legal/Contract Law/Commercial Contracts")
            st.caption("• Technology/Software/Cloud Computing")

        st.markdown("---")

        # Extraction Settings
        st.markdown("### 🎚️ Settings")

        threshold = st.slider(
            "Relevance Threshold",
            min_value=0,
            max_value=100,
            value=70,
            step=5,
            help="Minimum relevance score (0-100). Higher = more selective."
        )

        st.caption(f"Current setting: {threshold}")
        if threshold >= 80:
            st.caption("🔴 High precision - fewer terms")
        elif threshold >= 60:
            st.caption("🟡 Balanced - recommended")
        else:
            st.caption("🟢 High recall - more terms")

        st.markdown("---")

        # About
        with st.expander("ℹ️ About"):
            st.markdown("""
            **TermExtractor** is an AI-powered terminology extraction system.

            **Features:**
            - Hierarchical domain classification
            - Bilingual & monolingual extraction
            - Multiple export formats
            - Cost optimization with caching

            **Powered by:** Anthropic's Claude AI
            """)

    # Main content
    tab1, tab2, tab3 = st.tabs(["📤 Upload & Extract", "📊 Results", "📖 Help"])

    with tab1:
        st.markdown("## 📤 Upload Document")

        uploaded_file = st.file_uploader(
            "Choose a file",
            type=['txt', 'docx', 'pdf', 'html', 'htm', 'xml'],
            help="Supported formats: TXT, DOCX, PDF, HTML, XML"
        )

        if uploaded_file:
            st.success(f"✓ File uploaded: {uploaded_file.name}")

            # File info
            file_size = len(uploaded_file.getvalue())
            st.info(f"📊 File size: {file_size:,} bytes ({file_size/1024:.2f} KB)")

            # Extract button
            col1, col2, col3 = st.columns([1, 2, 1])

            with col2:
                if st.button("🚀 Extract Terms", use_container_width=True, type="primary"):
                    if not api_key_input:
                        st.error("❌ Please provide an API key in the sidebar")
                    else:
                        with st.spinner("🔄 Extracting terminology... This may take a few moments."):
                            try:
                                # Run extraction
                                result, usage = asyncio.run(extract_terms_async(
                                    uploaded_file.getvalue(),
                                    uploaded_file.name,
                                    source_lang,
                                    target_lang,
                                    domain_path,
                                    threshold,
                                    model,
                                    api_key_input
                                ))

                                # Store in session state
                                st.session_state.extraction_results = result
                                st.session_state.api_usage = usage

                                st.success("✅ Extraction completed successfully!")
                                st.balloons()

                            except Exception as e:
                                st.error(f"❌ Error during extraction: {str(e)}")
                                st.exception(e)
        else:
            st.info("👆 Upload a document to get started")

            # Example files suggestion
            with st.expander("💡 Don't have a file? Try these examples"):
                st.markdown("""
                Create a text file with content like:

                **Medical Example:**
                ```
                The patient presents with acute myocardial infarction.
                Administered thrombolytic therapy and beta-blockers.
                Monitoring cardiac enzymes and ECG continuously.
                ```

                **Technology Example:**
                ```
                Cloud computing leverages distributed systems for scalability.
                Microservices architecture enables independent deployment.
                Kubernetes orchestrates containerized applications.
                ```

                **Legal Example:**
                ```
                The parties agree to a non-disclosure obligation.
                Confidential information includes proprietary data.
                Breach of contract may result in liquidated damages.
                ```
                """)

    with tab2:
        st.markdown("## 📊 Extraction Results")

        if st.session_state.extraction_results:
            display_results(
                st.session_state.extraction_results,
                st.session_state.api_usage
            )
        else:
            st.info("👈 Extract terms from a document first")

            # Show placeholder visualization
            st.markdown("### Preview")
            st.markdown("Results will appear here after extraction, including:")

            col1, col2 = st.columns(2)

            with col1:
                st.markdown("""
                **Statistics:**
                - Total terms extracted
                - Relevance distribution
                - Domain classification
                - API usage metrics
                """)

            with col2:
                st.markdown("""
                **Export Options:**
                - Excel (.xlsx) with multiple sheets
                - CSV for spreadsheets
                - TBX for CAT tools
                - JSON for APIs
                """)

    with tab3:
        st.markdown("## 📖 Help & Documentation")

        # Quick Start
        with st.expander("🚀 Quick Start", expanded=True):
            st.markdown("""
            1. **Set API Key:** Enter your Anthropic API key in the sidebar
            2. **Configure:** Choose model, languages, and domain (optional)
            3. **Upload:** Upload a document (TXT, DOCX, PDF, HTML, XML)
            4. **Extract:** Click "Extract Terms" button
            5. **Review:** View results and download in your preferred format
            """)

        # Settings Guide
        with st.expander("⚙️ Settings Guide"):
            st.markdown("""
            **Model Selection:**
            - **Claude 3.5 Sonnet** (Recommended): Best balance of quality and cost
            - **Claude 3 Opus**: Highest quality, premium pricing
            - **Claude 3.5 Haiku**: Fastest and most economical

            **Relevance Threshold:**
            - **80-100**: High precision, only very relevant terms
            - **60-79**: Balanced, recommended for most use cases
            - **0-59**: High recall, includes more terms

            **Domain Path:**
            - Specify up to 5 levels: `Level1/Level2/Level3/Level4/Level5`
            - Leave empty for automatic detection
            - Examples:
              - `Medical/Healthcare/Veterinary Medicine`
              - `Technology/Software Development/API Design`
              - `Legal/Contract Law/Intellectual Property`
            """)

        # Supported Formats
        with st.expander("📁 Supported Formats"):
            col1, col2 = st.columns(2)

            with col1:
                st.markdown("""
                **Input Formats:**
                - Plain Text (.txt)
                - Microsoft Word (.docx)
                - PDF (.pdf)
                - HTML (.html, .htm)
                - XML (.xml)
                """)

            with col2:
                st.markdown("""
                **Export Formats:**
                - Excel (.xlsx) - Multiple sheets
                - CSV (.csv) - Spreadsheet compatible
                - TBX (.tbx) - CAT tools
                - JSON (.json) - API integration
                """)

        # Troubleshooting
        with st.expander("🔧 Troubleshooting"):
            st.markdown("""
            **API Key Issues:**
            - Make sure your API key is valid
            - Check that you have sufficient credits
            - API key is not stored, re-enter if needed

            **File Upload Issues:**
            - Maximum file size: 200MB
            - Ensure file is not corrupted
            - Try converting to TXT if issues persist

            **Low Quality Results:**
            - Try specifying a domain path
            - Increase relevance threshold
            - Use Claude 3 Opus for better quality

            **Cost Concerns:**
            - Use Claude 3.5 Haiku for lower costs
            - Enable caching (automatic)
            - Process smaller files first
            """)

        # Links
        with st.expander("🔗 Resources"):
            st.markdown("""
            - [GitHub Repository](https://github.com/yourusername/Termextractor)
            - [Documentation](https://termextractor.readthedocs.io)
            - [API Reference](https://github.com/yourusername/Termextractor/blob/main/docs/api.md)
            - [Examples](https://github.com/yourusername/Termextractor/blob/main/EXAMPLES.md)

            **Get API Key:**
            - [Anthropic Console](https://console.anthropic.com)
            """)

    # Footer
    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center; color: #666;'>
        <p>TermExtractor v1.0.0 | Powered by Anthropic's Claude AI |
        <a href='https://github.com/yourusername/Termextractor'>GitHub</a></p>
        </div>
        """,
        unsafe_allow_html=True
    )


if __name__ == "__main__":
    main()
