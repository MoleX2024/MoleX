from setuptools import setup, find_packages

setup(
    name="molex",
    version="0.0.1",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "scikit-learn",
        "datasets",
        "pandas",
        "sentence-transformers",
        "imodelsx",
    ],
    author="Molex Group",
    description="Offical package for Molex",
    long_description=open('README.md').read(),
    long_description_content_type="text/markdown",
    url="https://github.com/yourname/molex24",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
    ],
    python_requires='>=3.6',
)