from setuptools import find_namespace_packages
from setuptools import setup


def _get_version():
  with open('torch_toolkit/__init__.py') as fp:
    for line in fp:
      if line.startswith('__version__'):
        g = {}
        exec(line, g)  # pylint: disable=exec-used
        return g['__version__']
    raise ValueError('`__version__` not defined in `torch_toolkit/__init__.py`')


def _parse_requirements(requirements_txt_path):
  with open(requirements_txt_path) as fp:
    return fp.read().splitlines()


_VERSION = _get_version()

setup(
    name='torch_toolkit',
    version=_VERSION,
    url='https://github.com/DavidSlayback/torch-toolkit',
    license='Apache 2.0',
    author='David Slayback',
    description='My Pytorch toolkit. Minimal requiresments',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    author_email='slayback.d@northeastern.edu',
    # Contained modules and scripts.
    packages=find_namespace_packages(include=['torch_toolkit', 'torch_toolkit.*']),
    install_requires=_parse_requirements('requirements.txt'),
    requires_python='>=3.8',
    include_package_data=True,
    zip_safe=False,
    # PyPI package information.
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'Intended Audience :: Education',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: GNU',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Topic :: Scientific/Engineering :: Mathematics',
        'Topic :: Software Development :: Libraries :: Python Modules',
        'Topic :: Software Development :: Libraries',
    ],
)