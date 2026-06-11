from setuptools import setup, find_packages
from pathlib import Path

this_dir = Path(__file__).parent
long_description = (this_dir / "aim_cu" / "README.rst").read_text(encoding="utf-8")

setup(
    name="aim-cu",
    version="0.2.0",
    description="AIM-CU: A CUSUM-based tool for AI Monitoring",
    long_description=long_description,
    long_description_content_type="text/x-rst",
    authors="Smriti Prathapan, Berkman Sahiner, Dhaval Kadia and Ravi Samala",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=[
        "numpy>=1.23",
        "pandas>=1.5",
        "matplotlib>=3.6",
        # "rpy2>=3.5",
        "tomli; python_version<'3.11'",
    ],
    include_package_data=True,
    package_data={
        "aim_cu": ["config.toml", "SECURITY.md", "README.rst"],
    },
    url="https://github.com/DIDSR/AIM-CU",
    project_urls={
        "User Inteface" : "https://huggingface.co/spaces/didsr/AIM-CU",
    },
)