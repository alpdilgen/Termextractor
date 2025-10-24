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
    url="https://github.com/yourusername/Termextractor",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Topic :: Text Processing :: Linguistic",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.9",
    install_requires=[
        "anthropic>=0.34.0",
        "python-dotenv>=1.0.0",
        "pandas>=2.0.0",
        "numpy>=1.24.0",
        "openpyxl>=3.1.0",
        "spacy>=3.7.0",
        "nltk>=3.8.0",
        "fastapi>=0.109.0",
        "uvicorn[standard]>=0.27.0",
        "aiohttp>=3.9.0",
        "python-docx>=1.1.0",
        "PyPDF2>=3.0.0",
        "lxml>=5.0.0",
        "beautifulsoup4>=4.12.0",
        "cryptography>=42.0.0",
        "SQLAlchemy>=2.0.0",
        "tqdm>=4.66.0",
        "rich>=13.7.0",
        "click>=8.1.7",
        "pydantic>=2.5.0",
        "PyYAML>=6.0.1",
        "requests>=2.31.0",
        "loguru>=0.7.0",
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
            "plotly>=5.18.0",
            "dash>=2.14.0",
        ],
        "docs": [
            "sphinx>=7.2.0",
            "sphinx-rtd-theme>=2.0.0",
            "sphinx-autodoc-typehints>=1.25.0",
        ],
        "advanced-nlp": [
            "stanza>=1.8.0",
            "transformers>=4.35.0",
            "sentencepiece>=0.1.99",
        ],
    },
    entry_points={
        "console_scripts": [
            "termextractor=termextractor.ui.cli:main",
        ],
    },
    include_package_data=True,
    package_data={
        "termextractor": [
            "config/*.yaml",
            "data/*.json",
        ],
    },
    zip_safe=False,
)
