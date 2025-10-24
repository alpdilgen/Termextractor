"""Streamlit web application for TermExtractor."""

import streamlit as st
import asyncio
import tempfile
from pathlib import Path
import pandas as pd
from datetime import datetime
import os
import sys
import json # Added for JSON export

# Add src directory to Python path for Streamlit Cloud
# This needs to run before other project imports
src_path = Path(__file__).resolve().parent.parent # Use resolve() for robustness
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))
    print(f"Added {src_path} to sys.path") # Debug print

# FIXED: Corrected imports relative to src/, adjusted io->file_io
try:
    from api.anthropic_client import AnthropicClient, APIResponse # Added APIResponse
    from api.api_manager import APIManager, RateLimitConfig, CacheConfig
    from extraction.term_extractor import TermExtractor, ExtractionResult # Added ExtractionResult
    from core.progress_tracker import ProgressTracker
    from file_io.format_exporter import FormatExporter # Changed io -> file_io
    from utils.constants import ANTHROPIC_MODELS, SUPPORTED_LANGUAGES
    from utils.helpers import load_config
    print("All project imports successful.") # Debug print
except ImportError as e:
     print(f"Import Error: {e}")
     st.error(f"Internal Server Error: Could not load necessary components. Details: {e}")
     # Optional: Add more details or exit if critical components fail
     # sys.exit(1) # Uncomment to stop app if imports fail


# Page configuration (should be called only once)
st.set_page_config(
    page_title="TermExtractor - AI-Powered Terminology Extraction",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS
st.markdown("""
<style>
    /* ... (CSS rules seem okay) ... */
</style>
""", unsafe_allow_html=True)


# --- Session State ---
def init_session_state():
    """Initialize session state variables."""
    defaults = {
        'extraction_results': None,
        'api_usage': None,
        'processing': False,
        'api_key': os.getenv('ANTHROPIC_API_KEY', ''), # Load from env var initially
        'model_selection': ANTHROPIC_MODELS[0],
        'source_lang': list(SUPPORTED_LANGUAGES.keys())[0], # Default to first language
        'target_lang': list(SUPPORTED_LANGUAGES.keys())[1], # Default to second language
        'enable_bilingual': False,
        'domain_path_input': "",
        'enable_custom_domain': False,
        'threshold': 70,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

# --- Helper Functions ---
def get_api_key():
    """Get API key from session state."""
    return st.session_state.get('api_key', '')

# --- Async Extraction Logic ---
async def extract_terms_async(
    file_content,
    file_name,
    source_lang,
    target_lang, # This will be None if enable_bilingual is False
    domain_path, # This will be None if enable_custom_domain is False
    threshold,
    model,
    api_key
):
    """Async function to extract terms."""
    # Ensure a temporary directory exists
    temp_dir = Path(tempfile.gettempdir()) / "termextractor_streamlit"
    temp_dir.mkdir(parents=True, exist_ok=True)

    # Use a NamedTemporaryFile within the specific temp directory
    tmp_path = None # Initialize path variable
    try:
        # Save uploaded file temporarily
        # delete=False is important on some systems, ensure cleanup in finally
        with tempfile.NamedTemporaryFile(delete=False, suffix=Path(file_name).suffix, dir=temp_dir) as tmp_file:
            tmp_file.write(file_content)
            tmp_path = Path(tmp_file.name)
            st.write(f"DEBUG: Saved temp file to {tmp_path}") # Debug output

        # Initialize components
        # Consider loading config here if needed for RateLimitConfig etc.
        # config = load_config() # Assuming load_config finds default or handles errors
        # rate_config = RateLimitConfig(**config.get("api", {}).get("rate_limiting", {}))
        # cache_config = CacheConfig(**config.get("api", {}).get("caching", {}))
        # Hardcoding defaults for now if config isn't loaded
        rate_config = RateLimitConfig()
        cache_config = CacheConfig()

        client = AnthropicClient(api_key=api_key, model=model)
        api_manager = APIManager(client=client, rate_limit_config=rate_config, cache_config=cache_config)
        # ProgressTracker for Streamlit might need a different implementation
        # Using simple st.progress for now, tracker is more for CLI/batch
        # tracker = ProgressTracker(enable_rich_output=False)

        extractor = TermExtractor(
            api_client=api_manager,
            # progress_tracker=tracker, # Not directly used with st.progress
            default_relevance_threshold=threshold # Pass default, method uses specific
        )

        st.write("DEBUG: Starting extractor.extract_from_file...") # Debug output
        # Extract terms
        result = await extractor.extract_from_file(
            file_path=tmp_path,
            source_lang=source_lang,
            target_lang=target_lang, # Pass the potentially None value
            domain_path=domain_path, # Pass the potentially None value
            relevance_threshold=threshold # Pass the specific threshold
        )
        st.write(f"DEBUG: extract_from_file completed. Result terms count: {len(result.terms)}") # Debug output

        # Get usage stats
        usage = client.get_usage_stats()
        st.write(f"DEBUG: Got usage stats: {usage}") # Debug output

        return result, usage

    except FileNotFoundError as fnf_err:
         st.error(f"Error: Temporary file not found during processing: {fnf_err}")
         logger.error(f"FileNotFoundError during extraction: {fnf_err}", exc_info=True)
         return ExtractionResult(terms=[], source_language=source_lang, metadata={"error": str(fnf_err)}), {} # Return empty result
    except ValueError as val_err: # Catch parsing errors etc.
         st.error(f"Error during file processing: {val_err}")
         logger.error(f"ValueError during extraction: {val_err}", exc_info=True)
         return ExtractionResult(terms=[], source_language=source_lang, metadata={"error": str(val_err)}), {} # Return empty result
    except Exception as e:
        st.error(f"An unexpected error occurred during extraction: {e}")
        logger.error(f"Unexpected error during extraction: {e}", exc_info=True)
        # Return empty result with error details
        return ExtractionResult(terms=[], source_language=source_lang, metadata={"error": str(e)}), {}
    finally:
        # Clean up temp file
        if tmp_path and tmp_path.exists():
            try:
                tmp_path.unlink()
                st.write(f"DEBUG: Deleted temp file {tmp_path}") # Debug output
            except Exception as cleanup_err:
                st.warning(f"Could not delete temporary file {tmp_path}: {cleanup_err}")
                logger.warning(f"Failed to delete temp file {tmp_path}: {cleanup_err}")


# --- UI Components ---
def display_results(result: ExtractionResult, usage: Dict):
    """Display extraction results."""
    st.markdown("---")
    st.markdown("## 📊 Extraction Results")

    if not result or not hasattr(result, 'terms'):
        st.warning("No valid extraction results to display.")
        return

    # Statistics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Terms", len(result.terms))
    with col2:
        high_relevance = len(result.get_high_relevance_terms()) # Assumes method exists
        st.metric("High Relevance (≥80)", high_relevance)
    with col3:
        medium_relevance = len(result.get_medium_relevance_terms()) # Assumes method exists
        st.metric("Medium Relevance (60-79)", medium_relevance)
    with col4:
        low_relevance = len(result.get_low_relevance_terms()) # Assumes method exists
        st.metric("Low Relevance (<60)", low_relevance)

    # Domain information
    st.markdown("### 🎯 Domain Classification")
    domain_path_display = " → ".join(result.domain_hierarchy) if result.domain_hierarchy else "N/A"
    st.info(f"**Domain:** {domain_path_display}")

    # API Usage
    st.markdown("### 💰 API Usage")
    col1_api, col2_api, col3_api = st.columns(3)
    if usage: # Check if usage dict is not empty
        with col1_api:
            st.metric("Total Requests", usage.get('total_requests', 0))
        with col2_api:
            st.metric("Total Tokens", f"{usage.get('total_tokens', 0):,}")
        with col3_api:
            st.metric("Estimated Cost", f"${usage.get('estimated_cost', 0.0):.4f}")
    else:
        st.warning("API usage data not available.")


    # Terms table
    st.markdown("### 📝 Extracted Terms")
    filter_option = st.selectbox(
        "Filter by relevance:",
        ["All Terms", "High Relevance (≥80)", "Medium Relevance (60-79)", "Low Relevance (<60)"]
    )

    # Apply filtering based on selection
    if filter_option == "High Relevance (≥80)":
        filtered_terms = result.get_high_relevance_terms()
    elif filter_option == "Medium Relevance (60-79)":
        filtered_terms = result.get_medium_relevance_terms()
    elif filter_option == "Low Relevance (<60)":
        filtered_terms = result.get_low_relevance_terms()
    else:
        filtered_terms = result.terms

    if filtered_terms:
        # Convert list of Term objects to list of dicts for DataFrame
        df_data = [term.to_dict() for term in filtered_terms]
        df = pd.DataFrame(df_data)

        # Select and order columns for display
        display_columns = [
            "term", "translation", "domain", "subdomain", "pos", "definition",
            "context", "relevance_score", "confidence_score", "frequency"
        ]
        # Filter df to only include desired columns that actually exist
        df_display = df[[col for col in display_columns if col in df.columns]]

        st.dataframe(df_display, use_container_width=True, height=400) # use_container_width is deprecated, use width=None

        # Download options
        st.markdown("### 💾 Download Results")
        col1_dl, col2_dl, col3_dl, col4_dl = st.columns(4)

        # Prepare base filename
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        base_filename = f"terms_{timestamp}"

        # Get full DataFrame (unfiltered by relevance display) for downloads
        all_terms_data = [term.to_dict() for term in result.terms]
        df_all = pd.DataFrame(all_terms_data)
        df_all_export = df_all[[col for col in display_columns if col in df_all.columns]]


        with col1_dl:
            # Excel export
            # Use BytesIO for in-memory file generation
            from io import BytesIO
            excel_buffer = BytesIO()
            with pd.ExcelWriter(excel_buffer, engine='openpyxl') as writer:
                 df_all_export.to_excel(writer, sheet_name="All Terms", index=False)
                 # Optionally add filtered sheets
                 if result.get_high_relevance_terms():
                      df_high = pd.DataFrame([t.to_dict() for t in result.get_high_relevance_terms()])
                      df_high[[col for col in display_columns if col in df_high.columns]].to_excel(writer, sheet_name="High Relevance", index=False)
                 # Add metadata sheet
                 stats_data = result.statistics.copy()
                 stats_data["Source Language"] = result.source_language
                 stats_data["Target Language"] = result.target_language or "N/A"
                 stats_data["Domain Hierarchy"] = " → ".join(result.domain_hierarchy)
                 if usage:
                      stats_data.update({f"API_{k}": v for k, v in usage.items()})
                 stats_df = pd.DataFrame(list(stats_data.items()), columns=['Metric', 'Value'])
                 stats_df.to_excel(writer, sheet_name="Statistics", index=False)

            st.download_button(
                 label="📊 Download Excel",
                 data=excel_buffer.getvalue(),
                 file_name=f"{base_filename}.xlsx",
                 mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                 use_container_width=True # Deprecated
            )

        with col2_dl:
            # CSV export
            csv = df_all_export.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📄 Download CSV",
                data=csv,
                file_name=f"{base_filename}.csv",
                mime="text/csv",
                use_container_width=True # Deprecated
            )

        with col3_dl:
            # JSON export
            json_data = json.dumps(result.to_dict(), indent=2, ensure_ascii=False).encode('utf-8')
            st.download_button(
                 label="📋 Download JSON",
                 data=json_data,
                 file_name=f"{base_filename}.json",
                 mime="application/json",
                 use_container_width=True # Deprecated
            )

        with col4_dl:
            # TBX export - Requires async call within sync context, tricky in Streamlit
            # Option 1: Generate TBX in memory synchronously (if simple enough)
            # Option 2: Use a button + spinner that calls an async function via asyncio.run()
            # Option 3: Offload to a background task (complex)

            # Simplified approach (sync generation, might block UI for large files):
            if st.button("🗂️ Download TBX", use_container_width=True): # Deprecated use_container_width
                 with st.spinner("Generating TBX file..."):
                      try:
                           # This runs synchronously - might block UI
                           # In a real app, use threading or async properly if needed
                           from io import BytesIO
                           from lxml import etree # Ensure lxml is installed

                           # (TBX generation logic adapted from format_exporter._export_tbx)
                           NSMAP = {None: "urn:iso:std:iso:30042:ed-1"}
                           root = etree.Element("tbx", nsmap=NSMAP)
                           root.set("{http://www.w3.org/XML/1998/namespace}lang", result.source_language)
                           header = etree.SubElement(root, "tbxHeader")
                           # ... (add header details) ...
                           text_el = etree.SubElement(root, "text")
                           body = etree.SubElement(text_el, "body")
                           for term in result.terms:
                                termEntry = etree.SubElement(body, "termEntry")
                                # ... (populate termEntry as in format_exporter) ...

                           tbx_buffer = BytesIO()
                           tree = etree.ElementTree(root)
                           tree.write(tbx_buffer, pretty_print=True, xml_declaration=True, encoding="UTF-8")

                           st.download_button(
                                label="⬇️ Download TBX File", # Change label after generation
                                data=tbx_buffer.getvalue(),
                                file_name=f"{base_filename}.tbx",
                                mime="application/xml"
                           )
                           st.success("TBX ready for download!")
                      except ImportError:
                           st.error("lxml library needed for TBX export. Please install it.")
                      except Exception as tbx_e:
                           st.error(f"Failed to generate TBX: {tbx_e}")
                           logger.error(f"TBX generation failed: {tbx_e}", exc_info=True)


    else:
        st.warning("No terms match the selected filter.")


# --- Main App ---
def main():
    """Main Streamlit application."""
    init_session_state()

    # Header
    st.markdown('<div class="main-header">📚 TermExtractor</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">AI-Powered Terminology Extraction with Claude</div>', unsafe_allow_html=True)

    # --- Sidebar ---
    with st.sidebar:
        st.markdown("## ⚙️ Configuration")

        # API Key
        st.markdown("### 🔑 API Key")
        api_key_input = st.text_input(
            "Anthropic API Key", type="password", value=st.session_state.api_key,
            help="Enter your Anthropic API key. It will not be stored long-term unless enabled."
        )
        # Update session state immediately if changed
        if api_key_input != st.session_state.api_key:
             st.session_state.api_key = api_key_input
             st.rerun() # Rerun to reflect change and potentially remove warning

        if st.session_state.api_key:
            st.success("✓ API Key provided")
        else:
            st.warning("⚠️ API Key required for extraction")

        st.markdown("---")

        # Model Selection
        st.markdown("### 🤖 Model")
        st.session_state.model_selection = st.selectbox(
            "Select Claude Model", ANTHROPIC_MODELS,
            index=ANTHROPIC_MODELS.index(st.session_state.model_selection) if st.session_state.model_selection in ANTHROPIC_MODELS else 0,
            help="Claude 3.5 Sonnet recommended"
        )
        # ... (Model cost info) ...
        st.markdown("---")

        # Language Settings
        st.markdown("### 🌍 Languages")
        st.session_state.source_lang = st.selectbox(
            "Source Language", list(SUPPORTED_LANGUAGES.keys()),
            index=list(SUPPORTED_LANGUAGES.keys()).index(st.session_state.source_lang) if st.session_state.source_lang in SUPPORTED_LANGUAGES else 0,
            format_func=lambda x: f"{x.upper()} - {SUPPORTED_LANGUAGES[x]}"
        )
        st.session_state.enable_bilingual = st.checkbox("Bilingual Extraction", value=st.session_state.enable_bilingual)
        target_lang_to_pass = None
        if st.session_state.enable_bilingual:
            st.session_state.target_lang = st.selectbox(
                "Target Language", list(SUPPORTED_LANGUAGES.keys()),
                index=list(SUPPORTED_LANGUAGES.keys()).index(st.session_state.target_lang) if st.session_state.target_lang in SUPPORTED_LANGUAGES else 1,
                format_func=lambda x: f"{x.upper()} - {SUPPORTED_LANGUAGES[x]}"
            )
            target_lang_to_pass = st.session_state.target_lang # Set the variable to pass
        st.markdown("---")

        # Domain Settings
        st.markdown("### 🎯 Domain")
        st.session_state.enable_custom_domain = st.checkbox("Custom Domain Path", value=st.session_state.enable_custom_domain)
        domain_path_to_pass = None
        if st.session_state.enable_custom_domain:
            st.session_state.domain_path_input = st.text_input(
                "Domain Hierarchy", value=st.session_state.domain_path_input,
                placeholder="e.g., Medical/Healthcare/Cardiology",
                help="Use / to separate levels."
            )
            domain_path_to_pass = st.session_state.domain_path_input # Set variable to pass
        # ... (Domain examples) ...
        st.markdown("---")

        # Extraction Settings
        st.markdown("### 🎚️ Settings")
        st.session_state.threshold = st.slider(
            "Relevance Threshold", min_value=0, max_value=100,
            value=st.session_state.threshold, step=5,
            help="Minimum relevance score (0-100)."
        )
        # ... (Threshold captions) ...
        st.markdown("---")

        # About section
        with st.expander("ℹ️ About"):
             # ... (About text) ...
             pass

    # --- Main Content ---
    tab1, tab2, tab3 = st.tabs(["📤 Upload & Extract", "📊 Results", "📖 Help"])

    with tab1:
        st.markdown("## 📤 Upload Document")
        uploaded_file = st.file_uploader(
            "Choose a file",
            type=['txt', 'docx', 'pdf', 'html', 'htm', 'xml', 'xlf', 'xliff', 'sdlxliff', 'mqxliff'], # Added XLIFF types
            help="Supported formats: TXT, DOCX, PDF, HTML, XML, XLIFF variants"
        )

        if uploaded_file:
            st.success(f"✓ File uploaded: {uploaded_file.name}")
            file_size = len(uploaded_file.getvalue())
            st.info(f"📊 File size: {file_size:,} bytes ({file_size/1024:.2f} KB)")

            col1_btn, col2_btn, col3_btn = st.columns([1, 2, 1])
            with col2_btn:
                # Disable button if processing or no API key
                disable_extract = st.session_state.processing or not st.session_state.api_key
                if st.button("🚀 Extract Terms", use_container_width=True, type="primary", disabled=disable_extract): # Deprecated use_container_width
                    if not st.session_state.api_key:
                        st.error("❌ Please provide an API key in the sidebar")
                    else:
                        st.session_state.processing = True
                        st.session_state.extraction_results = None # Clear previous results
                        st.session_state.api_usage = None
                        st.rerun() # Rerun to show spinner immediately

        # --- Processing Logic ---
        if st.session_state.processing:
            if uploaded_file: # Check again inside processing block
                 with st.spinner("🔄 Extracting terminology... This may take a few moments."):
                    try:
                        # Ensure API key is still valid (it might have been cleared)
                        current_api_key = get_api_key()
                        if not current_api_key:
                             raise ValueError("API Key is missing.")

                        # Run extraction
                        result, usage = asyncio.run(extract_terms_async(
                            file_content=uploaded_file.getvalue(),
                            file_name=uploaded_file.name,
                            source_lang=st.session_state.source_lang,
                            target_lang=target_lang_to_pass, # Use the variable set in sidebar logic
                            domain_path=domain_path_to_pass, # Use the variable set in sidebar logic
                            threshold=st.session_state.threshold,
                            model=st.session_state.model_selection,
                            api_key=current_api_key
                        ))

                        # Store results ONLY if successful extraction
                        if result and (not hasattr(result, 'metadata') or "error" not in result.metadata):
                             st.session_state.extraction_results = result
                             st.session_state.api_usage = usage
                             st.success("✅ Extraction completed successfully!")
                             st.balloons()
                        elif result and hasattr(result, 'metadata') and "error" in result.metadata:
                             # Error already shown by extract_terms_async
                             st.error(f"Extraction failed: {result.metadata['error']}")
                        else:
                             st.error("Extraction process did not return expected results.")


                    except ValueError as ve: # Catch specific errors like missing key
                         st.error(f"❌ Configuration Error: {ve}")
                         logger.error(f"ValueError before/during extraction call: {ve}", exc_info=True)
                    except Exception as e:
                        st.error(f"❌ An unexpected error occurred: {e}")
                        logger.error(f"Unexpected error in extraction button logic: {e}", exc_info=True)
                        st.exception(e) # Show full traceback in UI for debugging
                    finally:
                        st.session_state.processing = False # Mark processing as done
                        st.rerun() # Rerun to update UI (remove spinner, show results/errors)
            else:
                 st.warning("File was uploaded but seems unavailable now. Please re-upload.")
                 st.session_state.processing = False # Stop processing if file disappears
                 st.rerun()

        elif not uploaded_file: # Only show these if not processing and no file uploaded
            st.info("👆 Upload a document to get started")
            with st.expander("💡 Don't have a file? Try these examples"):
                 # ... (Example texts) ...
                 pass

    with tab2:
        st.markdown("## 📊 Extraction Results")
        if st.session_state.extraction_results:
            display_results(st.session_state.extraction_results, st.session_state.api_usage)
        elif st.session_state.processing:
             st.info("Processing... Results will appear here shortly.")
        else:
            st.info("👈 Extract terms from a document first in the 'Upload & Extract' tab.")
            # ... (Placeholder info) ...

    with tab3:
        st.markdown("## 📖 Help & Documentation")
        # ... (Help expanders) ...
        pass

    # --- Footer ---
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
    # Ensure event loop policy is set correctly for Streamlit+Asyncio on some systems
    # asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy()) # Example for Windows
    main()
