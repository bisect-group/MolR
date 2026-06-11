from pathlib import Path

from setuptools import setup, find_packages

ROOT = Path(__file__).parent

requirements = (ROOT / 'requirements.txt').read_text().splitlines()
long_description = (ROOT / 'README.md').read_text()

setup(
    name='MolR',
    version='0.1.0',
    author='Hongwei Wang, Weijiang Li, Xiaomeng Jin, Kyunghyun Cho, Heng Ji, Jiawei Han, Martin Burke',
    author_email='{hongweiw@illinois.edu',
    description='Chemical-Reaction-Aware Molecule Representation Learning',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/hwwang55/MolR',
    packages=find_packages(include=['MolR', 'MolR.*']),
    include_package_data=True,
    install_requires=requirements,
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.8',
    package_data={
        'MolR': ['models/**/*']
    },
)
