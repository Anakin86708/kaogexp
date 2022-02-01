from setuptools import setup

with open("requirements.txt", "r") as fh:
    requirements = fh.readlines()

setup(
    name='kaogexp',
    version='0.2',
    packages=['kaogexp'],
    author='Ariel Tadeu da Silva',
    author_email='silva.ariel@icloud.com',
    description='Created to explain instances from ML models.',
    install_requires=[req.strip() for req in requirements if req[:2] != "# "],
)
