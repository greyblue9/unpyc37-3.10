unpyc.unpyc3
============

Decompiler for Python 3.7 +
- Decent support for Python 3.8
- Mostly complete support for Python 3.9
- Minimal support for Python 3.10 (forked from https://github.com/figment/unpyc3)

The original project is unsupported and only works for earlier versions of Python. This one only aims to work with Python 3.7.

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
`python3.8 -m unpyc.unpyc3 file.pyc`

To decompile instructions 61 to end:
`python3.8 -m unpyc.unpyc3 file.pyc 61`

