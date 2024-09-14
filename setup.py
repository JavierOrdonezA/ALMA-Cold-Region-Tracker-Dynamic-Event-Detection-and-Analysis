"""This module is used for the setup of the 'Alma tracking' package.

It handles the package setup including reading long description from README.md,
installing dependencies from requirements.txt, and defining package metadata.
"""

from setuptools import find_packages, setup

# Reading the contents of the README.md file
with open('README.md', 'r', encoding='utf-8') as fh:
    long_description = fh.read()

# Reading the dependencies from requirements.txt
with open('requirements.txt') as f:
    requirements = f.read().splitlines()

setup(
    name='AlmaTracking',
    version='0.1.0',
    author='F. Javier Ordonez - Araujo',
    author_email='',
    description='A dynamic repository for streamlined Alma tracking.',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='',
    packages=find_packages(),
    install_requires=requirements,
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
)
