import sys
from unpyc import unpyc3
sys.modules.update(
  {
    "unpyc3": unpyc3
  }
)
from unpyc.unpyc3 import *
