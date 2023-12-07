from setuptools import setup, find_packages
import os

required = [
    "numpy",
    "matplotlib",
    "seaborn",
    "networkx",
    "keras",
    "scikit-learn",
    "tensorflow",
    "markupsafe",
    "brian2"
]

setup(
    packages=find_packages(where="src"),
    # packages=['sim_soens_lite'],
    package_dir={"sim_soens_lite":"src/sim_soens_lite"},
    package_data={'sim_soens_lite' :["sim_soens_lite/soen_sim_data/*.soen"]},
    include_package_data=True,
    install_requires=required,
    )