from setuptools import setup

setup(
    name='kaogexp',
    version='0.1',
    packages=['kaogexp'],
    package_dir={'': 'test'},
    url='',
    license='',
    author='Ariel Tadeu da Silva',
    author_email='silva.ariel2013@gmail.com',
    description='Created to explain instances from ML models.',
    install_requires=[
        'pandas',
        'numpy',
        'distython',
        'sklearn',
        'torch',
        'kaog',
        'lhsmdu',
        'requests',
        'seaborn',
        'matplotlib',
        'setuptools'
    ],
)
