from setuptools import find_packages, setup

setup(
    name="pytorch_trainer",
    version="0.0.1",
    packages=find_packages(),
    install_requires=["torch==2.1.1@https://download.pytorch.org/whl/cu118", 
                      "torchvision==0.16.1@https://download.pytorch.org/whl/cu118"],
)