from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="agentic-cli-coder",
    version="1.0.2",
    description="Open source, cheap, and premier coding agent",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=find_packages(),
    entry_points={
        "console_scripts": [
            "agentic=agentic.cli:main",
        ],
    },
    author="",
    license="ISC",
    install_requires=[
        "chromadb",
        "cryptography",
        "litellm",
        "pathspec",
        "playwright",
        "prompt-toolkit>=3.0",
        "python-dotenv",
        "requests",
        "rich",
        "simple-term-menu",
        "tiktoken",
    ],
    extras_require={
        'test': ['pytest', 'pytest-mock'],
    },
)
