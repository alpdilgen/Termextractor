#!/bin/bash

# TermExtractor Streamlit Web App Launcher
# This script launches the Streamlit web interface

echo "🚀 Starting TermExtractor Web Interface..."
echo ""

# Check if virtual environment is activated
if [[ -z "$VIRTUAL_ENV" ]]; then
    echo "⚠️  Warning: No virtual environment detected."
    echo "It's recommended to activate your virtual environment first:"
    echo "  source venv/bin/activate"
    echo ""
fi

# Check if ANTHROPIC_API_KEY is set
if [[ -z "$ANTHROPIC_API_KEY" ]]; then
    echo "⚠️  Warning: ANTHROPIC_API_KEY environment variable not set."
    echo "You can either:"
    echo "  1. Set it now: export ANTHROPIC_API_KEY='your-key-here'"
    echo "  2. Enter it in the web interface when it opens"
    echo ""
fi

# Check if streamlit is installed
if ! python -c "import streamlit" 2>/dev/null; then
    echo "❌ Streamlit not found. Installing..."
    pip install streamlit plotly
    echo ""
fi

# Launch Streamlit
echo "✨ Opening TermExtractor in your browser..."
echo ""
echo "📝 Note: You can stop the server anytime with Ctrl+C"
echo ""

streamlit run src/ui/streamlit_app.py \
    --server.port 8501 \
    --server.address localhost \
    --server.headless false \
    --browser.gatherUsageStats false
