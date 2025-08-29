"""Setup script for transformers package."""
from setuptools import setup, find_packages

setup(
    name="transformers-clean-arch",
    version="1.0.0",
    description="Transformer training with Clean Architecture",
    author="Craftsman Developer",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.11",
    install_requires=[
        "torch>=2.0.0",
        "pytorch-lightning>=2.0.0",
        "transformers>=4.0.0",
        "datasets>=2.0.0",
        "pydantic>=2.0.0",
        "pyyaml>=6.0",
    ],
    extras_require={
        "dev": [
            "pytest>=8.0.0",
            "pytest-cov>=6.0.0",
            "black>=23.0.0",
            "flake8>=6.0.0",
            "mypy>=1.0.0",
            "isort>=5.12.0",
        ],
    },
)
