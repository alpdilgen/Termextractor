"""Streamlit web application for TermExtractor."""

import streamlit as st
import asyncio
import tempfile
from pathlib import Path
import pandas as pd
from datetime import datetime
import os
import sys
import json
from io import BytesIO # For in-memory file generation
# --- Added Optional and Tuple ---
from typing import Any, Dict, List, Optional, Union, Tuple
from lxml import etree # For TBX export

# --- Add src directory to Python path ---
src_path = Path(__file__).resolve().parent.parent
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))
    print(f"DEBUG: Added {src_path} to sys.path")

# --- Project Imports ---
try:
    from api.anthropic_client import AnthropicClient, APIResponse
    from api.api_manager import APIManager, RateLimitConfig, CacheConfig
    from extraction import TermExtractor, ExtractionResult, Term # Import data models via extraction package
    from core.progress_tracker import ProgressTracker
    from file_io.format_exporter import FormatExporter
    from utils.constants import ANTHROPIC_MODELS, SUPPORTED_LANGUAGES
    from utils.helpers import load_config
    from loguru import logger
    print("DEBUG: All project imports successful.")
except ImportError as e:
     error_message = f"Internal Server Error: Could not load necessary components. Details: {e}. Check logs."
     print(f"FATAL IMPORT ERROR: {error_message}")
     # Basic config for error page, might fail if st itself isn't loaded
     try: st.set_page_config(page_title="Error", layout="centered")
     except: pass
     st.error(error_message)
     st.stop()
except Exception as e:
     error_message = f"An unexpected error occurred during startup: {e}. Check logs."
     print(f"FATAL STARTUP ERROR: {error_message}")
     try: st.set_page_config(page_title="Error", layout="centered")
     except: pass
     st.error(error_message)
     st.stop()


# --- Page Configuration ---
st.set_page_config(
    page_title="TermExtractor - AI Terminology Extraction",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- Custom CSS ---
st.markdown("""
<style>
    /* ... (Your CSS rules) ... */
</style>
""", unsafe_allow_html=True)

# --- Session State Initialization ---
def init_session_state():
    """Initialize session state variables if they don't exist."""
    defaults = {
        'extraction_results': None,
        'api_usage': None,
        'processing': False,
        'api_key': os.getenv('ANTHROPIC_API_KEY', ''),
        'model_selection': ANTHROPIC_MODELS[0] if ANTHROPIC_MODELS else "default-model",
        'source_lang': list(SUPPORTED_LANGUAGES.keys())[0] if SUPPORTED_LANGUAGES else "en",
        'target_lang': list(SUPPORTED_LANGUAGES.keys())[1] if len(SUPPORTED_LANGUAGES) > 1 else "de",
        'enable_bilingual': False,
        'domain_path_input': "",
        'enable_custom_domain': False,
        'threshold': 70.0,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

# --- Helper Functions ---
def get_api_key():
    """Get API key safely from session state."""
    return st.session_state.get('api_key', '')

# --- Async Extraction Logic ---
async def extract_terms_async(
    file_content: bytes,
    file_name: str,
    source_lang: str,
    target_lang: Optional[str], # Type hint uses Optional
    domain_path: Optional[str], # Type hint uses Optional
    threshold: float,
    model: str,
    api_key: str
) -> Tuple[ExtractionResult, Dict]: # Type hint uses Tuple and Dict
    """Async function to extract terms, handling initialization and cleanup."""
    temp_dir = Path(tempfile.gettempdir()) / "termextractor_streamlit"
    temp_dir.mkdir(parents=True, exist_ok=True)
    tmp_path: Optional[Path] = None # Type hint uses Optional

    empty_result = ExtractionResult(terms=[], source_language=source_lang, target_language=target_lang)
    empty_usage = {}

    try:
        suffix = Path(file_name).suffix or ".tmp"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix, dir=temp_dir, mode='wb') as tmp_file:
            tmp_file.write(file_content)
            tmp_path = Path(tmp_file.name)
        logger.info(f"Saved temp file for processing: {tmp_path}")

        rate_config = RateLimitConfig()
        cache_config = CacheConfig()
        client = AnthropicClient(api_key=api_key, model=model)
        api_manager = APIManager(client=client, rate_limit_config=rate_config, cache_config=cache_config)
        extractor = TermExtractor(api_client=api_manager, default_relevance_threshold=threshold)

        logger.info(f"Starting extraction for file: {file_name}")
        result = await extractor.extract_from_file(
            file_path=tmp_path,
            source_lang=source_lang,
            target_lang=target_lang,
            domain_path=domain_path if domain_path else None,
            relevance_threshold=threshold
        )
        logger.info(f"Extraction completed for {file_name}. Terms found (before filter in result object): {len(result.terms)}")
        usage = client.get_usage_stats()
        logger.info(f"API Usage for {file_name}: {usage}")

        return result, usage

    except FileNotFoundError as fnf_err:
         logger.error(f"FileNotFoundError during extraction: {fnf_err}", exc_info=True)
         empty_result.metadata = {"error": f"Internal error: Temporary file vanished - {fnf_err}"}
         return empty_result, empty_usage
    except ValueError as val_err:
         logger.error(f"ValueError during extraction setup or parsing: {val_err}", exc_info=True)
         empty_result.metadata = {"error": f"Processing error: {val_err}"}
         return empty_result, empty_usage
    except Exception as e:
        logger.error(f"Unexpected error during extraction process: {e}", exc_info=True)
        empty_result.metadata = {"error": f"Unexpected error: {e}"}
        return empty_result, empty_usage
    finally:
        if tmp_path and tmp_path.exists():
            try:
                tmp_path.unlink()
                logger.info(f"Deleted temp file {tmp_path}")
            except Exception as cleanup_err:
                logger.warning(f"Failed to delete temp file {tmp_path}: {cleanup_err}")

# --- UI Components ---
def display_results(result: ExtractionResult, usage: Dict): # Type hint uses ExtractionResult
    """Display extraction results in Streamlit."""
    st.markdown("---")
    st.markdown("## 📊 Extraction Results")

    if not isinstance(result, ExtractionResult):
        st.error("An internal error occurred: Invalid result format received.")
        return
    if "error" in result.metadata:
        st.error(f"Extraction failed: {result.metadata['error']}")
        return

    # --- Statistics ---
    stats = result.statistics
    col1, col2, col3, col4 = st.columns(4)
    with col1: st.metric("Total Terms", stats.get("total_terms", len(result.terms)))
    with col2: st.metric("High Relevance (≥80)", stats.get("high_relevance", 0))
    with col3: st.metric("Medium Relevance (60-79)", stats.get("medium_relevance", 0))
    with col4: st.metric("Low Relevance (<60)", stats.get("low_relevance", 0))

    # --- Domain ---
    st.markdown("### 🎯 Domain Classification")
    domain_path_display = " → ".join(result.domain_hierarchy) if result.domain_hierarchy else "N/A"
    st.info(f"**Domain:** {domain_path_display}")

    # --- API Usage ---
    st.markdown("### 💰 API Usage")
    col1_api, col2_api, col3_api = st.columns(3)
    if usage:
        with col1_api: st.metric("Total Requests", usage.get('total_requests', 'N/A'))
        with col2_api: st.metric("Total Tokens", f"{usage.get('total_tokens', 0):,}")
        with col3_api: st.metric("Estimated Cost", f"${usage.get('estimated_cost', 0.0):.4f}")
    else:
        st.warning("API usage data not available.")

    # --- Terms Table ---
    st.markdown("### 📝 Extracted Terms")
    if not result.terms:
        st.info("No terms were extracted matching the criteria.")
        return

    filter_option = st.selectbox(
        "Filter by relevance:",
        ["All Terms", "High Relevance (≥80)", "Medium Relevance (60-79)", "Low Relevance (<60)"]
    )

    if filter_option == "High Relevance (≥80)": filtered_terms = result.get_high_relevance_terms()
    elif filter_option == "Medium Relevance (60-79)": filtered_terms = result.get_medium_relevance_terms()
    elif filter_option == "Low Relevance (<60)": filtered_terms = result.get_low_relevance_terms()
    else: filtered_terms = result.terms

    if filtered_terms:
        df_data = [term.to_dict() for term in filtered_terms]
        df = pd.DataFrame(df_data)
        display_columns = [
            "term", "translation", "relevance_score", "confidence_score", "pos",
            "domain", "subdomain", "definition", "context", "frequency"
        ]
        df_display = df[[col for col in display_columns if col in df.columns]]
        st.dataframe(df_display, width=None, height=400) # Use width=None for container width

        # --- Download Buttons ---
        st.markdown("### 💾 Download Results")
        col1_dl, col2_dl, col3_dl, col4_dl = st.columns(4)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        base_filename = f"terms_{timestamp}"
        all_terms_data = [term.to_dict() for term in result.terms]
        df_all = pd.DataFrame(all_terms_data)
        export_columns = [
            "term", "translation", "domain", "subdomain", "pos", "definition",
            "context", "relevance_score", "confidence_score", "frequency",
            "is_compound", "is_abbreviation", "variants", "related_terms"
        ]
        df_all_export = df_all[[col for col in export_columns if col in df_all.columns]].copy()
        for col in ['variants', 'related_terms']:
             if col in df_all_export.columns:
                  df_all_export[col] = df_all_export[col].apply(lambda x: '; '.join(map(str, x)) if isinstance(x, list) else x)

        # -- Excel --
        with col1_dl:
            excel_buffer = BytesIO()
            # ... (Excel generation logic - seems okay) ...
            try:
                with pd.ExcelWriter(excel_buffer, engine='openpyxl') as writer:
                    df_all_export.to_excel(writer, sheet_name="All Terms", index=False)
                    high_terms = result.get_high_relevance_terms()
                    if high_terms:
                         df_high = pd.DataFrame([t.to_dict() for t in high_terms])
                         df_high[[col for col in export_columns if col in df_high.columns]].to_excel(writer, sheet_name="High Relevance", index=False)
                    stats_data = result.statistics.copy(); stats_data["Source Language"] = result.source_language; stats_data["Target Language"] = result.target_language or "N/A"; stats_data["Domain Hierarchy"] = " → ".join(result.domain_hierarchy) if result.domain_hierarchy else "N/A";
                    if usage: stats_data.update({f"API_{k}": v for k, v in usage.items()})
                    stats_df = pd.DataFrame(list(stats_data.items()), columns=['Metric', 'Value']); stats_df.to_excel(writer, sheet_name="Statistics", index=False)
                st.download_button(label="📊 Download Excel", data=excel_buffer.getvalue(), file_name=f"{base_filename}.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", key="dl_excel", use_container_width=True) # Deprecated
            except ImportError: st.error("Excel export requires `pandas` and `openpyxl`.")
            except Exception as ex_e: st.error(f"Excel export failed: {ex_e}"); logger.error(f"Excel export failed: {ex_e}", exc_info=True)

        # -- CSV --
        with col2_dl:
            # ... (CSV generation logic - seems okay) ...
             try:
                csv = df_all_export.to_csv(index=False).encode('utf-8')
                st.download_button(label="📄 Download CSV", data=csv, file_name=f"{base_filename}.csv", mime="text/csv", key="dl_csv", use_container_width=True) # Deprecated
             except Exception as csv_e: st.error(f"CSV export failed: {csv_e}"); logger.error(f"CSV export failed: {csv_e}", exc_info=True)

        # -- JSON --
        with col3_dl:
            # ... (JSON generation logic - seems okay) ...
             try:
                json_data = json.dumps(result.to_dict(), indent=2, ensure_ascii=False).encode('utf-8')
                st.download_button(label="📋 Download JSON", data=json_data, file_name=f"{base_filename}.json", mime="application/json", key="dl_json", use_container_width=True) # Deprecated
             except Exception as json_e: st.error(f"JSON export failed: {json_e}"); logger.error(f"JSON export failed: {json_e}", exc_info=True)


        # -- TBX --
        with col4_dl:
            # Generate TBX data in memory (synchronously)
            try:
                # (TBX generation logic)
                XML_LANG = "{http://www.w3.org/XML/1998/namespace}lang"; root = etree.Element("martif", type="TBX", attrib={XML_LANG: result.source_language or 'en'}); header = etree.SubElement(root, "martifHeader"); fileDesc = etree.SubElement(header, "fileDesc"); sourceDesc = etree.SubElement(fileDesc, "sourceDesc"); etree.SubElement(sourceDesc, "p").text = f"Generated by TermExtractor on {datetime.now().isoformat()}"; encodingDesc = etree.SubElement(header, "encodingDesc"); etree.SubElement(encodingDesc, "p", type="XCSURI").text = "TBX-Basic.xcs"; text_el = etree.SubElement(root, "text"); body = etree.SubElement(text_el, "body")
                for term_obj in result.terms:
                    termEntry = etree.SubElement(body, "termEntry"); descripGrp_concept = etree.SubElement(termEntry, "descripGrp"); etree.SubElement(descripGrp_concept, "descrip", type="subjectField").text = term_obj.domain or "General";
                    if term_obj.definition: etree.SubElement(descripGrp_concept, "descrip", type="definition").text = term_obj.definition
                    langSet_source = etree.SubElement(termEntry, "langSet", attrib={XML_LANG: result.source_language}); tig_source = etree.SubElement(langSet_source, "tig"); etree.SubElement(tig_source, "term").text = term_obj.term; etree.SubElement(tig_source, "termNote", type="partOfSpeech").text = term_obj.pos or "unknown"
                    if term_obj.context: etree.SubElement(tig_source, "descrip", type="context").text = term_obj.context
                    adminGrp_source = etree.SubElement(tig_source, "adminGrp"); etree.SubElement(adminGrp_source, "admin", type="termExtractor-relevanceScore").text = str(term_obj.relevance_score)
                    if term_obj.translation and result.target_language:
                        langSet_target = etree.SubElement(termEntry, "langSet", attrib={XML_LANG: result.target_language}); tig_target = etree.SubElement(langSet_target, "tig"); etree.SubElement(tig_target, "term").text = term_obj.translation
                tbx_buffer = BytesIO(); tree = etree.ElementTree(root); tree.write(tbx_buffer, pretty_print=True, xml_declaration=True, encoding="UTF-8")

                st.download_button(
                    label="🗂️ Download TBX", data=tbx_buffer.getvalue(),
                    file_name=f"{base_filename}.tbx", mime="application/xml",
                    key="dl_tbx", use_container_width=True # Deprecated
                )
            except ImportError: st.error("TBX export requires `lxml`.")
            except Exception as tbx_e: st.error(f"TBX export failed: {tbx_e}"); logger.error(f"TBX export failed: {tbx_e}", exc_info=True)
    else:
        st.warning("No terms to display or download based on the selected filter.")


# --- Main App Execution ---
def main():
    """Main Streamlit application function."""
    init_session_state() # Ensure state is initialized first

    # Header
    st.markdown('<div class="main-header">📚 TermExtractor</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">AI-Powered Terminology Extraction with Claude</div>', unsafe_allow_html=True)

    # --- Sidebar ---
    with st.sidebar:
        st.markdown("## ⚙️ Configuration")
        # --- API Key Input ---
        st.markdown("### 🔑 API Key")
        api_key_input = st.text_input("Anthropic API Key", type="password", key="api_key_widget", help="Enter key. Stored in session state only.")
        # Update state immediately if input changes
        if api_key_input != st.session_state.api_key: st.session_state.api_key = api_key_input
        if st.session_state.api_key: st.success("✓ API Key provided (in session)")
        else: st.warning("⚠️ API Key required")
        st.markdown("---")

        # --- Model Selection ---
        st.markdown("### 🤖 Model")
        model_options = ANTHROPIC_MODELS
        try: current_model_index = model_options.index(st.session_state.model_selection)
        except ValueError: current_model_index = 0; st.session_state.model_selection = model_options[0]
        st.session_state.model_selection = st.selectbox("Select Claude Model", model_options, index=current_model_index, help="Claude 3.5 Sonnet recommended")
        st.markdown("---")

        # --- Language Settings ---
        st.markdown("### 🌍 Languages")
        lang_options = list(SUPPORTED_LANGUAGES.keys())
        try: current_source_index = lang_options.index(st.session_state.source_lang)
        except ValueError: current_source_index = 0; st.session_state.source_lang = lang_options[0]
        st.session_state.source_lang = st.selectbox("Source Language", lang_options, index=current_source_index, format_func=lambda x: f"{x.upper()} - {SUPPORTED_LANGUAGES[x]}")

        st.session_state.enable_bilingual = st.checkbox("Bilingual Extraction", key="enable_bilingual_widget")
        target_lang_to_pass = None
        if st.session_state.enable_bilingual:
             try: current_target_index = lang_options.index(st.session_state.target_lang)
             except ValueError: current_target_index = 1 if len(lang_options)>1 else 0; st.session_state.target_lang = lang_options[current_target_index]
             st.session_state.target_lang = st.selectbox("Target Language", lang_options, index=current_target_index, format_func=lambda x: f"{x.upper()} - {SUPPORTED_LANGUAGES[x]}")
             target_lang_to_pass = st.session_state.target_lang
        st.markdown("---")

        # --- Domain Settings ---
        st.markdown("### 🎯 Domain")
        st.session_state.enable_custom_domain = st.checkbox("Custom Domain Path", key="enable_custom_domain_widget")
        domain_path_to_pass = None
        if st.session_state.enable_custom_domain:
            st.session_state.domain_path_input = st.text_input("Domain Hierarchy", key="domain_path_widget", placeholder="e.g., Medical/Healthcare", help="Use / to separate levels.")
            if st.session_state.domain_path_input.strip(): domain_path_to_pass = st.session_state.domain_path_input.strip()
        st.markdown("---")

        # --- Extraction Settings ---
        st.markdown("### 🎚️ Settings")
        st.session_state.threshold = st.slider("Relevance Threshold", min_value=0.0, max_value=100.0, value=float(st.session_state.threshold), step=5.0, key="threshold_widget", help="Minimum relevance score (0-100).")
        st.markdown("---")

        # --- About Section ---
        with st.expander("ℹ️ About"): st.markdown("TermExtractor v1.0...")

    # --- Main Area Tabs ---
    tab1, tab2, tab3 = st.tabs(["📤 Upload & Extract", "📊 Results", "📖 Help"])

    with tab1:
        st.markdown("## 📤 Upload Document")
        uploaded_file = st.file_uploader(
            "Choose a file",
            type=['txt', 'docx', 'pdf', 'html', 'htm', 'xml', 'xlf', 'xliff', 'sdlxliff', 'mqxliff'],
            help="Supports TXT, DOCX, PDF, HTML, XML, XLIFF variants",
            key="file_uploader_widget"
        )

        if uploaded_file is not None:
            st.success(f"✓ File selected: {uploaded_file.name} ({uploaded_file.size/1024:.2f} KB)")
            col1_btn, col2_btn, col3_btn = st.columns([1, 1.5, 1])
            with col2_btn:
                disable_extract = st.session_state.processing or not st.session_state.api_key
                if st.button("🚀 Extract Terms", key="extract_button", type="primary", disabled=disable_extract, use_container_width=True): # Deprecated
                    if not st.session_state.api_key: st.error("❌ API key required.")
                    else:
                        st.session_state.processing = True; st.session_state.extraction_results = None; st.session_state.api_usage = None
                        st.rerun() # Show spinner immediately

        # --- Processing Logic ---
        if st.session_state.processing:
            uploaded_file_state = st.session_state.get("file_uploader_widget")
            if uploaded_file_state is not None:
                 progress_area = st.empty() # Placeholder for progress bar/text
                 progress_area.progress(0.0, text="Initializing...")
                 try:
                    current_api_key = get_api_key()
                    if not current_api_key: raise ValueError("API Key is missing.")
                    progress_area.progress(0.1, text="Extracting terms (this may take time)...")

                    # Run async function
                    result, usage = asyncio.run(extract_terms_async(
                        file_content=uploaded_file_state.getvalue(), file_name=uploaded_file_state.name,
                        source_lang=st.session_state.source_lang, target_lang=target_lang_to_pass,
                        domain_path=domain_path_to_pass, threshold=st.session_state.threshold,
                        model=st.session_state.model_selection, api_key=current_api_key
                    ))
                    progress_area.progress(0.9, text="Processing results...")

                    # Store results if successful
                    if isinstance(result, ExtractionResult) and "error" not in result.metadata:
                         st.session_state.extraction_results = result
                         st.session_state.api_usage = usage
                         progress_area.progress(1.0, text="Completed!")
                         st.success("✅ Extraction completed!")
                         st.balloons()
                    elif isinstance(result, ExtractionResult) and "error" in result.metadata:
                         st.error(f"Extraction failed: {result.metadata['error']}")
                         progress_area.progress(1.0, text="Failed!")
                    else:
                         st.error("Extraction process returned unexpected data.")
                         progress_area.progress(1.0, text="Failed!")

                 except ValueError as ve: st.error(f"❌ Configuration/Input Error: {ve}"); logger.error(f"ValueError: {ve}", exc_info=True)
                 except Exception as e: st.error(f"❌ Unexpected Error: {e}"); logger.error(f"Unexpected Error: {e}", exc_info=True); st.exception(e)
                 finally:
                    progress_area.empty() # Remove progress bar
                    st.session_state.processing = False
                    # Optionally clear uploader: st.session_state.file_uploader_widget = None
                    st.rerun() # Update UI
            else:
                 st.warning("File missing. Please re-upload."); st.session_state.processing = False; st.rerun()

        elif not uploaded_file: # Show only if not processing and no file
            st.info("👆 Upload a document to get started.")
            # ... (Example texts expander) ...

    with tab2:
        st.markdown("## 📊 Extraction Results")
        if st.session_state.extraction_results and not st.session_state.processing:
            display_results(st.session_state.extraction_results, st.session_state.api_usage)
        elif st.session_state.processing: st.info("Processing...")
        else: st.info("👈 Extract terms using the 'Upload & Extract' tab.")

    with tab3:
        st.markdown("## 📖 Help & Documentation")
        # ... (Help expanders) ...
        pass

    # --- Footer ---
    st.markdown("---")
    st.markdown(
        """<div style='text-align: center; color: #666;'>
        <p>TermExtractor v1.0.0 | Powered by Anthropic Claude |
        <a href='https://github.com/yourusername/Termextractor'>GitHub</a></p>
        </div>""", unsafe_allow_html=True
    )


# --- Run App ---
if __name__ == "__main__":
    logger.info("Starting Streamlit App")
    main()
