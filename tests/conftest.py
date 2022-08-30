from pathlib import Path
import sys

sys.path.append(
  (Path(__file__).parent / "src").absolute().as_posix()
)
sys.path.append(
  (Path(__file__).parent / "tests" / "unpyc").absolute().as_posix()
)

