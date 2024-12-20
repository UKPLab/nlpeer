from setuptools import setup, find_packages
setup(
    name='nlpeer',
    version='0.2',
    author="Ubiquitous Knowledge Processing Lab",
    author_email="nils.dycke@tu-darmstadt.de",
    description="Code utilities for loading NLPeer data and for running experiments.",
    long_description="README.md",
    packages=find_packages(),
    python_requires='>=3.10',
)