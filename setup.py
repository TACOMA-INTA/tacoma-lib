from setuptools import setup

# To use a consistent encoding
from codecs import open
from os import path

# The directory containing this file
HERE = path.abspath(path.dirname(__file__))

# Get the long description from the README file
with open(path.join(HERE, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

with open(path.join(HERE, 'requirements.txt'), encoding='utf-8') as f:
    requirements = f.read().splitlines()



setup(
    name="Tacoma lib",
    version="1.0.0",
    description="TACOMA lib",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author='Jaime Bowen Varela, Rodrigo Castellanos, Alejandro Gorgues',
    author_email="jbowvar@inta.es, rcasgar@inta.es, gorguesva@inta.es", 
    packages=["tacoma"],
    include_package_data=True,
    install_requires= [
        "pandas>=1.3.5",
        "numpy>=1.21.6",
        "scipy>=1.4.1", # mainly to work with Google Colab
        "scikit-learn>=1.0.2",
        "matplotlib>=3.5.0"
    ],
)
