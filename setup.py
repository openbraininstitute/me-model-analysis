"""Setup."""
import os
from setuptools import setup, find_packages


setup(
    name='me_model_analysis',
    description='ME-Model Analysis',
    version=os.environ['VERSION'],
    author='Blue Brain Project, EPFL',
    install_requires=('uvicorn[standard]', 'fastapi', 'boto3', 'requests'),
    # bluepyemodel & bluepyemodelnexus & nexus-forge will be installed in Dockerfile
    packages=find_packages(exclude=[]),
    scripts=[],
)
