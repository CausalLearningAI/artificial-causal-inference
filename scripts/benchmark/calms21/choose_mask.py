"""Apply the PRE-REGISTERED mask choice of eci_res.sh to OUT/mask/maskqa.json and write plan/mask_rule.txt:
the variant with the highest mean keypoint recall (resident and intruder averaged) among the variants keeping <= 25% of
the video cells; a variant within 0.01 recall of a simpler one loses to it (order: dark, dark_bright, absdiff,
absdiff_floor). Login safe (reads one small JSON)."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from calms_common import OUT, PLAN  # noqa: E402

ORDER = ['dark', 'dark_bright', 'absdiff', 'absdiff_floor']
Q = json.loads((OUT / 'mask' / 'maskqa.json').read_text())
rec = {r: (Q[r]['kp_recall_resident'] + Q[r]['kp_recall_intruder']) / 2 for r in ORDER}
ok = [r for r in ORDER if Q[r]['kept_frac_of_video_cells'] <= 0.25]
assert ok, 'no variant keeps <= 25% of the video cells'
best = max(ok, key=rec.get)
for r in ok:  # simplest variant within 0.01 of the best
    if rec[r] >= rec[best] - 0.01:
        best = r
        break
txt = f"{best}\n# chosen by choose_mask.py (rule in eci_res.sh); recall {json.dumps({r: round(v, 4) for r, v in rec.items()})}; " \
      f"kept frac {json.dumps({r: round(Q[r]['kept_frac_of_video_cells'], 4) for r in ORDER})}\n"
(PLAN / 'mask_rule.txt').write_text(txt)
print(txt)
