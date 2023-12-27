from setuptools import find_packages, setup

setup(
    name="pytorch_trainer",
    version="0.0.1",
    packages=find_packages(),
    install_requires=["pytorch==2.1.1", "torchvision==0.16.1", "pytorch-cuda=11.8"],
)