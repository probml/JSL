from setuptools import find_packages, setup

setup(
    name="jsl",
    packages=find_packages(),
    install_requires=[
        "chex",
        "dataclasses",
        "jaxlib",
        "jax",
        "tensorflow_probability"
    ]
)
