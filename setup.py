from setuptools import find_packages
from skbuild import setup

setup(
    name="planning",
    version="0.0.1",
    packages=find_packages(where="planning", exclude=["planning.*"]),
    package_dir={"": "planning"},
    cmake_install_dir="planning/planning",
    include_package_data=True,
    python_requires=">=3.8",
)
