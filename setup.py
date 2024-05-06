#  This code is a part of the Democratising Archival X-ray Astronomy (DAXA) module.
#  Last modified by David J Turner (turne540@msu.edu) 25/04/2024, 18:11. Copyright (c) The Contributors

from os import path

from setuptools import setup, find_packages

# Uses the README as the long description
this_directory = path.abspath(path.dirname(__file__))
with open(path.join(this_directory, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='daxa',
    version='{{VERSION_PLACEHOLDER}}',
    packages=find_packages(),
    url='https://github.com/DavidT3/DAXA',
    license='BSD 3',
    author='David J Turner',
    author_email='turne540@msu.edu',
    description='Democratising Astronomy X-ray Archives (DAXA) is an easy-to-use Python module which enables '
                'the simple processing and reduction of archives of X-ray telescope observations.',
    long_description=long_description,
    long_description_content_type='text/markdown',
    setup_requires=[],
    install_requires=["numpy==1.24.4", "astroquery==0.4.7", "pandas==2.0.3", "astropy==5.2.2", "packaging==21.3",
                      "tqdm==4.66.2", "exceptiongroup==1.0.4", "scipy==1.10.1", "tabulate==0.9.0", "unlzw3==0.2.2"],
    include_package_data=True,
    python_requires='>=3.8'
)
