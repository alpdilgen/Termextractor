# Deploying TermExtractor to Streamlit Cloud

## Quick Deploy

### Option 1: Automatic Deployment (Recommended)

1. **Fork/Push to GitHub**
   ```bash
   git push origin claude/advanced-terminology-extraction-011CURbVSmKpHuRd7VWVptKh
   ```

2. **Go to Streamlit Cloud**
   - Visit https://share.streamlit.io
   - Click "New app"
   - Select your repository: `alpdilgen/Termextractor`
   - Branch: `claude/advanced-terminology-extraction-011CURbVSmKpHuRd7VWVptKh`
   - Main file path: `src/ui/streamlit_app.py`

3. **Advanced Settings**
   - Click "Advanced settings"
   - Python version: `3.11`
   - Add to "Secrets":
     ```toml
     ANTHROPIC_API_KEY = "your-api-key-here"
     ```
   - Note: requirements.txt is used automatically (no selection needed)

4. **Deploy**
   - Click "Deploy"
   - Wait for deployment (2-3 minutes)

### Option 2: Full Development Setup (Local Only)

Use `requirements-dev.txt` for local development with all features:

```bash
pip install -r requirements-dev.txt
```

## Files for Streamlit Cloud

The repository includes two requirement files:

- **requirements.txt** - Minimal dependencies for Streamlit Cloud (used automatically)
- **requirements-dev.txt** - Full dependencies for local development with all features

## Troubleshooting Deployment Errors

### Error: "installer returned a non-zero exit code"

**Solution 1:** Ensure `requirements.txt` contains minimal dependencies (already fixed in this repo)

**Solution 2:** If still failing, create `.streamlit/config.toml` in repo:
```toml
[server]
maxUploadSize = 200
enableCORS = false
```

### Error: Missing dependencies during runtime

The streamlined requirements might miss some optional features. The app is designed to handle missing dependencies gracefully.

**What works with minimal requirements:**
- ✅ File upload (TXT, DOCX, PDF, HTML, XML)
- ✅ Term extraction
- ✅ All export formats (XLSX, CSV, TBX, JSON)
- ✅ Domain classification
- ✅ All UI features

**Optional features (require full install):**
- ⚠️ Some advanced NLP features (not used in web UI)
- ⚠️ Development tools (not needed for deployment)

### Error: Build timeout

**Solution:** Streamlit Cloud has build time limits. Use `requirements-streamlit.txt` which installs faster.

### Error: Memory issues

**Solution:** Streamlit Cloud free tier has memory limits. The streamlined version uses less memory.

## Configuration

### Setting API Key

**Option 1: Streamlit Cloud Secrets (Recommended)**
```toml
# In Streamlit Cloud Secrets
ANTHROPIC_API_KEY = "sk-ant-..."
```

**Option 2: User Input**
Users can enter API key in the sidebar (not stored, session only)

### Custom Domain (Optional)

In Streamlit Cloud settings:
- Go to Settings → Custom domain
- Add your domain
- Update DNS records as instructed

## Environment Variables

Supported environment variables:

```bash
ANTHROPIC_API_KEY=your-key-here
LOG_LEVEL=INFO
```

## Local Testing Before Deploy

Test with Streamlit Cloud requirements:

```bash
# Create test environment
python -m venv test_env
source test_env/bin/activate

# Install minimal requirements (same as Streamlit Cloud uses)
pip install -r requirements.txt

# Test locally
streamlit run src/ui/streamlit_app.py

# If it works locally, it should work on Streamlit Cloud
```

## Alternative Deployment Options

### Docker Deployment

Create `Dockerfile`:
```dockerfile
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

EXPOSE 8501

CMD ["streamlit", "run", "src/ui/streamlit_app.py"]
```

Build and run:
```bash
docker build -t termextractor .
docker run -p 8501:8501 -e ANTHROPIC_API_KEY=your-key termextractor
```

### Heroku Deployment

1. Create `Procfile`:
```
web: streamlit run src/ui/streamlit_app.py --server.port=$PORT
```

2. Create `runtime.txt`:
```
python-3.11.0
```

3. Deploy:
```bash
heroku create your-app-name
git push heroku claude/advanced-terminology-extraction-011CURbVSmKpHuRd7VWVptKh:main
```

### Railway Deployment

1. Connect GitHub repository
2. Set build command: `pip install -r requirements.txt`
3. Set start command: `streamlit run src/ui/streamlit_app.py`
4. Add environment variable: `ANTHROPIC_API_KEY`

## Performance Optimization

For Streamlit Cloud:

1. **Enable Caching** (already implemented)
   - API responses are cached
   - File parsing is cached

2. **Resource Limits**
   - Free tier: 1 GB RAM
   - Paid tier: More resources available

3. **Cost Management**
   - Set daily API cost limits in the app
   - Monitor usage in the UI

## Monitoring

After deployment, monitor:
- App logs in Streamlit Cloud dashboard
- API usage in Anthropic console
- User feedback

## Support

If deployment issues persist:

1. Check Streamlit Cloud logs
2. Verify all files are pushed to GitHub
3. Ensure `requirements-streamlit.txt` is used
4. Check Python version (3.11 recommended)
5. Open an issue on GitHub with error logs

## Success Checklist

✅ Repository pushed to GitHub
✅ Branch: `claude/advanced-terminology-extraction-011CURbVSmKpHuRd7VWVptKh`
✅ File exists: `requirements.txt` (minimal deps)
✅ File exists: `src/ui/streamlit_app.py`
✅ File exists: `packages.txt` (empty - no system deps needed)
✅ API key set in Streamlit Secrets
✅ Python version set to 3.11
✅ Deploy clicked

## Example Streamlit Cloud Configuration

```toml
# Secrets (in Streamlit Cloud UI)
ANTHROPIC_API_KEY = "sk-ant-api03-..."

# Advanced Settings in UI
Python version: 3.11
# Note: requirements.txt is automatically detected
App URL: https://your-app-name.streamlit.app
```

Your app will be live at: `https://[your-app-name].streamlit.app`
