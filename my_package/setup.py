from setuptools import find_packages, setup

with open("Readme.md", "r") as f:
    long_description = f.read()

setup(
    name="my_package_ramakant9687",
    version="0.0.1",
    author="RAMAKANT",
    author_email="ramakant9687@gmail.com",
    description="A small package to work with House Prediction Dataset",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/ramakant9687/Package_building",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
