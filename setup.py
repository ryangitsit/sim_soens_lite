from setuptools import setup, find_packages

setup(
    packages=find_packages(where="src"),
    # packages=['sim_soens'],
    package_dir={"sim_soens":"src/sim_soens"},
    package_data={'sim_soens' :["sim_soens/soen_sim_data/*.soen"]},
    include_package_data=True,
    )