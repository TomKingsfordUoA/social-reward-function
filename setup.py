from setuptools import setup, find_packages  # type: ignore


__version__ = '0.3.0'

with open('requirements.txt') as f_requirements:
    requirements = f_requirements.read().splitlines()

setup(
    name='social-reward-function',
    version=__version__,
    author='Tom Kingsford',
    author_email='tkin063@aucklanduni.ac.nz',
    url='https://github.com/TomKingsfordUoA/social-reward-function',
    packages=find_packages(),
    entry_points={
        'console_scripts': ['srf=social_reward_function.cli:main']
    },
    include_package_data=True,
    install_requires=requirements,
    python_requires='>=3.7,<3.8',
)
