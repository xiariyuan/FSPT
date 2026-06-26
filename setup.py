from setuptools import find_packages, setup


setup(
    name="fspt",
    version="0.0.0",
    description="FSPT foundation package",
    packages=find_packages(include=("fspt", "fspt.*")),
    include_package_data=True,
)
