# small-SAE test status
- prereg commit c8f1b77 (scripts/eci/v2/v21_small_prereg.md), code scripts/eci/v2/v21_small.py, submit v21_small_submit.sh
- job ids: myids.txt (1702087-1702120); logs: small/logs/
- after both boot jobs finish: run `python scripts/eci/v2/v21_small.py report` in a CPU job (srun -p defaultp --exclude=eta310 -c 4 --mem=32G), then commit results/vision/eci_v2_1/small (git add -f, no logs of big size)
