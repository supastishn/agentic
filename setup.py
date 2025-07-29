from setuptools import setup, find_packages

setup(
    name="agentic",
    version="0.0.1",
    description="Open source, cheap, and premier coding agent",
    packages=find_packages(),
    entry_points={
        "console_scripts": [
            "agentic=agentic.cli:main",
        ],
    },
    author="",
    license="ISC",
    install_requires=[
        "litellm",
        "python-dotenv",
        "cryptography",
        "rich",
        "prompt-toolkit",
    ],
)
