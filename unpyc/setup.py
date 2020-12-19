#!/system/bin/python3

from distutils.core import setup


try:
    from distutils.command.build_py import build_py_2to3 as build_py
except ImportError:
    from distutils.command.build_py import build_py

try:
    from distutils.command.build_scripts import build_scripts_2to3 as build_scripts
except ImportError:
    from distutils.command.build_scripts import build_scripts

setup(
  name = "unpyc37",
  version = "0.0.00001",
  description = "The unpyc37 python package",
  author = "Unknown",
  license = "public domain",
  
    packages = [],
    scripts  = ['unpyc3.py'],

  cmdclass = { 'build_py': build_py, 'build_scripts': build_scripts }
)


