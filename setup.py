from setuptools import find_packages, setup

setup(
    name="jsl",
    packages=find_packages(),
    install_requires=[
        "jaxlib",
        "jax"
    ]
)
