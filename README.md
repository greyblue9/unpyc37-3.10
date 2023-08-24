unpyc.unpyc3
============

A decompiler for Python 3.7+

- This version is a maintained fork by [greyblue9](greyblue9@gmail.com).

- This version is based on [andrew-tavera/unpyc37](https://github.com/andrew-tavera/unpyc37),
- which is based on [figment/unpyc3](https://github.com/figment/unpyc3),
- which is based om [the original code, written by  arnodel](https://code.google.com/archive/p/unpyc3/).
- **Special thanks to `ADHD` who has helped a lot in improving and motivating me to work on it.**

## Features
- Good support for Python 3.7
- Decent support for Python 3.8
- Mostly complete support for Python 3.9
- Minimal support for Python 3.10 (forked from [figment/unpyc](https://github.com/figment/unpyc3))

The original project is sadly no longer maintained.

## Install

    python -m pip install .

## Usage
### Synopsis
`python3 -m unpyc.unpyc3 FILE.pyc [start [end]]`
### Options
    FILE.pyc  is the path to the
              file to decompile
    start     is  0 by default
    end       is -1 by default

### Examples
To decompile a whole file:
`python3.8 -m unpyc.unpyc3 file.pyc`

To decompile instructions 0-60:
`python3.8 -m unpyc.unpyc3 file.pyc 0 60`

To decompile instructions 61 to end:
`python3.8 -m unpyc.unpyc3 file.pyc 61`

### Background
The aim is to be able to recreate Python3 source code 
from code objects.

Current version is able to decompile itself 
successfully :). It has been tested with Python3.2 [3.7+]
only.

It currently reconstructs most of Python 3 constructs 
but probably needs to be tested more thoroughly. 
All feedback welcome.

#### Example:
```py

from unpyc.unpyc3 import decompile
def foo(x, y, z=3, *args):
  ...
  global g
  ...
  for i, j in zip(x, y):
    ...
    if z == i + j or args[i] == j:
      ...
      g = i, j
  ...
  return ..
print(decompile(foo))
```

~~Unpyc3 is made of a single python module.~~
Download unpyc3.py and try it now!

The unpyc3 module is able de decompile itself!
(try import unpyc3; unpyc3.decompile(unpyc3)) 
so theorically I could just distribute the .pyc file.

TODO:
* Support for keyword-only arguments
* Handle assert statements
* Show docstrings for functions and modules
* Nice spacing between function/class declarations
