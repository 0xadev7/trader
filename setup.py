"""Setup script for the trading bot."""
from setuptools import setup, find_packages

setup(
    name="ai-trading-bot",
    version="1.0.0",
    description="Advanced AI Trading Bot for Gate.io",
    author="Your Name",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.24.3",
        "pandas>=2.0.3",
        "python-dotenv>=1.0.0",
        "gate-api>=7.2.0",
        "torch>=2.1.0",
        "scikit-learn>=1.3.0",
        "ta>=0.10.2",
        "requests>=2.31.0",
        "loguru>=0.7.2",
    ],
    python_requires=">=3.8",
)

