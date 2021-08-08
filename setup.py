from setuptools import setup  # type: ignore


__version__ = '0.1.0'

with open('requirements.txt') as f_requirements:
    requirements = f_requirements.read().splitlines()

setup(
    name='social-robotics-reward',
    version=__version__,
    author='Tom Kingsford',
    author_email='tkin063@aucklanduni.ac.nz',
    packages=[
        'social_robotics_reward',
        'emotion_recognition_using_speech',
        'residual_masking_network',
    ],
    scripts=['social_robotics_reward/srr.py'],
    package_data={'social_robotics_reward': ['py.typed']},
    install_requires=requirements,
)
