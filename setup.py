import sys, re, os

try:
    from skbuild import setup
    import nanobind
except ImportError:
    raise

setup(
    name="planning",
    version="0.0.1",
    packages=["planning"],
    package_dir={"": "planning"},
    cmake_install_dir="planning/planning",
    include_package_data=True,
    python_requires=">=3.8",
)
