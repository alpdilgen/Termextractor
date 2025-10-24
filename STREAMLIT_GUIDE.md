# Running TermExtractor with Streamlit

## Quick Start

### Linux/Mac:

```bash
# Make sure you're in the project directory
cd Termextractor

# Activate virtual environment
source venv/bin/activate

# (Optional) Set API key
export ANTHROPIC_API_KEY='your-api-key-here'

# Launch Streamlit
./run_streamlit.sh
```

Or directly:
```bash
streamlit run src/ui/streamlit_app.py
```

### Windows:

```cmd
# Make sure you're in the project directory
cd Termextractor

# Activate virtual environment
venv\Scripts\activate

# (Optional) Set API key
set ANTHROPIC_API_KEY=your-api-key-here

# Launch Streamlit
run_streamlit.bat
```

Or directly:
```cmd
streamlit run src/ui/streamlit_app.py
```

## Using the Web Interface

Once Streamlit starts, it will automatically open in your browser at `http://localhost:8501`

### Main Features:

1. **Configuration Sidebar**
   - Enter API key (required)
   - Select Claude model
   - Choose source/target languages
   - Set custom domain path
   - Adjust relevance threshold

2. **Upload & Extract Tab**
   - Upload documents (TXT, DOCX, PDF, HTML, XML)
   - View file information
   - Extract terms with one click

3. **Results Tab**
   - View extraction statistics
   - See all extracted terms in a table
   - Filter by relevance (high/medium/low)
   - Download results in multiple formats:
     - Excel (.xlsx) with multiple sheets
     - CSV for spreadsheets
     - TBX for CAT tools
     - JSON for APIs

4. **Help Tab**
   - Quick start guide
   - Settings explanations
   - Supported formats
   - Troubleshooting tips

## Advanced Configuration

### Custom Port

Run on a different port:
```bash
streamlit run src/ui/streamlit_app.py --server.port 8080
```

### Remote Access

Allow connections from other machines:
```bash
streamlit run src/ui/streamlit_app.py --server.address 0.0.0.0
```

### Headless Mode

Run without automatically opening browser:
```bash
streamlit run src/ui/streamlit_app.py --server.headless true
```

## Streamlit Configuration File

Create `.streamlit/config.toml` for persistent settings:

```toml
[server]
port = 8501
address = "localhost"
headless = false

[browser]
gatherUsageStats = false

[theme]
primaryColor = "#1f77b4"
backgroundColor = "#ffffff"
secondaryBackgroundColor = "#f0f2f6"
textColor = "#262730"
font = "sans serif"
```

## Troubleshooting

### Port Already in Use

If port 8501 is busy:
```bash
streamlit run src/ui/streamlit_app.py --server.port 8502
```

### Streamlit Not Found

Install Streamlit:
```bash
pip install streamlit plotly
```

### API Key Issues

You can:
1. Set environment variable before launching
2. Enter in the sidebar when app is running (not stored)
3. Add to `.env` file in project root

### Import Errors

Make sure TermExtractor is installed:
```bash
pip install -e .
```

### File Upload Size Limit

By default, Streamlit limits uploads to 200MB. To change:

Create `.streamlit/config.toml`:
```toml
[server]
maxUploadSize = 500  # MB
```

## Tips for Best Experience

1. **Use Chrome/Firefox** for best compatibility
2. **Keep the terminal open** while using the app
3. **Don't close the browser tab** if you want to keep session state
4. **API key is not stored** - you'll need to re-enter on page refresh
5. **Cache is enabled** - repeated extractions are faster
6. **Results persist** - switch between tabs without losing data

## Stopping the Server

Press `Ctrl+C` in the terminal where Streamlit is running.

## Development Mode

For development with auto-reload:
```bash
streamlit run src/ui/streamlit_app.py --server.runOnSave true
```

## Screenshots

The interface includes:
- 📚 Clean, professional UI
- 🎨 Color-coded metrics
- 📊 Interactive data tables
- 💾 Multiple export formats
- 📖 Built-in help documentation
- 🎯 Real-time statistics
- 💰 Cost tracking

## Next Steps

After launching:
1. Enter your Anthropic API key in the sidebar
2. Configure your preferences (model, languages, etc.)
3. Upload a document
4. Click "Extract Terms"
5. Review and download results

Enjoy using TermExtractor! 🚀
