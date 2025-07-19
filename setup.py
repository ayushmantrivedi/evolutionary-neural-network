from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="evolutionary-neural-network",
    version="1.0.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="A revolutionary neural network that uses evolutionary algorithms instead of backpropagation",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/evolutionary-neural-network",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Information Analysis",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.7",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=6.0",
            "pytest-cov>=2.0",
            "black>=21.0",
            "flake8>=3.8",
            "mypy>=0.800",
        ],
        "docs": [
            "sphinx>=4.0",
            "sphinx-rtd-theme>=1.0",
            "myst-parser>=0.15",
        ],
    },
    entry_points={
        "console_scripts": [
            "evonet=hope:main",
        ],
    },
    include_package_data=True,
    package_data={
        "": ["*.md", "*.txt", "*.json", "*.csv"],
    },
    keywords=[
        "neural-network",
        "evolutionary-algorithm",
        "machine-learning",
        "artificial-intelligence",
        "genetic-algorithm",
        "no-backpropagation",
        "population-based",
        "classification",
        "regression",
    ],
    project_urls={
        "Bug Reports": "https://github.com/yourusername/evolutionary-neural-network/issues",
        "Source": "https://github.com/yourusername/evolutionary-neural-network",
        "Documentation": "https://github.com/yourusername/evolutionary-neural-network#readme",
    },
)
