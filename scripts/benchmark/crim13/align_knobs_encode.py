"""
CRIM13 SAE-knob test (align_knobs.py) - token-store rebuild. Runs align_encode.cmd_encode UNCHANGED except:
    SAVE = ['d3l_640_L20'] only (the d2l_560_L20 store is not needed by the knob arms; the DINOv2-B mask encoder and
    the lag-50 / random-cell gathers of the existing code still run, so the d3l_640_L20 arrays are bit-for-bit what the
    original encode wrote), and the encode JSON copy goes to OUT/knobs/encode_json/ (the original OUT/encode_json/
    records of the pre-registered test are not overwritten). Same arguments as align_encode.py.
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import align_encode as AE  # noqa: E402

AE.SAVE = ['d3l_640_L20']
AE.OUT = AE.OUT / 'knobs'

if __name__ == '__main__':
    AE.main()
