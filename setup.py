import os
from setuptools import setup


def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()


packname = 'scoter'

subpacknames = ['scoter.ie']


setup(
    name=packname,
    version='0.1.0',
    description='SCOTER: Multiple-event location by using static '
                'and source-specific station correction terms',
    long_description=read('README.md'),
    url='https://git.gfz-potsdam.de/nooshiri/scoter',
    author='Nima Nooshiri',
    author_email='nima.nooshiri@gfz-potsdam.de; '
                 'nima.nooshiri@gmail.com',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Science/Research',
        'Operating System :: POSIX :: Linux',
        'Programming Language :: Python :: 2.7',
        'Topic :: Scientific/Engineering'],
    keywords='seismology seismic-relocation station-terms',
    python_requires='>=2.7, <3',
    install_requires=[
        'numpy>=1.13.1',
        'scipy>=0.19.1',
        'pyrocko>=2017.11.22',
        'pyyaml>=3.11',
        'progressbar2>=3.34.2',
        'shapely>=1.6.4'],
    packages=[packname] + subpacknames,
    package_dir={'scoter': 'src'},
    scripts=['apps/scoter'],
    package_data={
        packname: [
            'data/config.sf',
            'data/FlinnEngdahl_seismic.pickle']})
