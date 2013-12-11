#!/usr/bin/env python

from distutils.core import setup, Extension

setup(name = 'jkutils',
      version = '1.1',
      description = 'Some utility functions',
      author = 'Jonas Kalderstam',
      author_email = 'jonas@kalderstam.se',
      url = 'https://github.com/spacecowboy/',
      packages = ['jkutils'],
      package_dir = {'jkutils': 'jkutils'},
      install_requires = ['scipy>=0.13',
                          'numpy>=1.7',
                          'matplotlib>=1.3'],
     )
