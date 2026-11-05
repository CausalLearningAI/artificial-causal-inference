"""ECI v2.6 sheets = v25_sheet with the v2.6 paths / arms (v26_eval patches); output results/vision/eci_v2_6/figures/."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import v26_eval  # noqa: E402,F401
import v25_sheet  # noqa: E402

if __name__ == '__main__':
    v25_sheet.main()
