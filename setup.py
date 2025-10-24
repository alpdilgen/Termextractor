"""Setup configuration for TermExtractor."""

from setuptools import setup, find_packages
from pathlib import Path

# Read README for long description
readme_path = Path(__file__).parent / "README.md"
long_description = readme_path.read_text(encoding="utf-8") if readme_path.exists() else ""

setup(
    name="termextractor",
    version="1.0.0",
    author="TermExtractor Team",
    author_email="contact@termextractor.com",
    description="State-of-the-art terminology extraction system powered by Anthropic's Claude AI",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/Termextractor", # Replace with your actual URL
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Topic :: Text Processing :: Linguistic",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License", # Choose your actual license
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.9",
    install_requires=[ # These should generally mirror requirements.txt
        "anthropic>=0.34.0",
        "python-dotenv>=1.0.0",
        "pandas>=2.0.0",
        "numpy>=1.24.0",
        "openpyxl>=3.1.0",
        "spacy>=3.7.0", # Note: spacy and nltk might be optional unless used heavily
        "nltk>=3.8.0",
        "fastapi>=0.109.0", # If you have an API component
        "uvicorn[standard]>=0.27.0", # If you have an API component
        "aiohttp>=3.9.0", # Check if actually used
        "python-docx>=1.1.0",
        "docx2txt", # Added
        "PyPDF2>=3.0.0",
        "lxml>=5.0.0",
        "beautifulsoup4>=4.12.0",
        "cryptography>=42.0.0",
        "SQLAlchemy>=2.0.0", # If using database features
        "tqdm>=4.66.0", # For CLI progress
        "rich>=13.7.0", # For CLI progress/output
        "click>=8.1.7", # For CLI
        "pydantic>=2.5.0",
        "PyYAML>=6.0.1",
        "requests>=2.31.0", # Check if actually used directly
        "loguru>=0.7.0",
        "keyring", # Added
        "python-dateutil>=2.8.2", # Added from requirements.txt
        # Add striprtf if used
    ],
    extras_require={
        "dev": [
            "pytest>=8.0.0",
            "pytest-asyncio>=0.23.0",
            "pytest-cov>=4.1.0",
            "black>=24.0.0",
            "flake8>=7.0.0",
            "mypy>=1.8.0",
            "pylint>=3.0.0",
            "isort>=5.13.0",
        ],
        "web": [
            "streamlit>=1.31.0",
            "plotly>=5.18.0", # Check if used
            "dash>=2.14.0", # Check if used
        ],
        "docs": [
            "sphinx>=7.2.0",
            "sphinx-rtd-theme>=2.0.0",
            "sphinx-autodoc-typehints>=1.25.0",
        ],
        "advanced-nlp": [
            "spacy>=3.7.0", # Moved main dependency to install_requires
            "nltk>=3.8.0", # Moved main dependency to install_requires
            "stanza>=1.8.0",
            "transformers>=4.35.0",
            "sentencepiece>=0.1.99",
            # Ensure spacy language models are handled (e.g., download instructions)
        ],
    },
    entry_points={
        "console_scripts": [
            "termextractor=ui.cli:main", # FIXED: Removed termextractor. prefix
        ],
    },
    include_package_data=True,
    package_data={
         # FIXED: Replace "termextractor" with the actual package containing config/data
         # If config/data are in src/core/, use "core":
        "core": [
            "config/*.yaml",
            "data/*.json",
        ],
         # If they are elsewhere, adjust accordingly.
         # If directly under src/, this method might not work; consider MANIFEST.in
    },
    zip_safe=False,
)
