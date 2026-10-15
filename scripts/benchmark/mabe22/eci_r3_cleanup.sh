#!/bin/bash
# MABe22 ECI round 3 cleanup (CPU job, after every experiment was scored, 2026-10-15 ~12:50): deletes the round-3 token
# stores (NFS tok_r3/ 56.2 GB / 420 files; BeeGFS adapted-encoder store 7.5 GB / 80 files) and moves the regenerable
# eval codes (r3/codes, 6.4 GB) and E4 firing bits (r3/e4 *.npy / *.npz, 3.2 GB) off the full NFS volume to BeeGFS.
#SBATCH --job-name=r3_cleanup
#SBATCH --output=results/vision/benchmark_mabe22/eci/r3/logs/%x_%j.out
set -euo pipefail
cd /nfs/scistore19/locatgrp/rcadei/artificial-causal-inference
E=results/vision/benchmark_mabe22/eci
B=/mnt/beegfs/locatgrp/shared/rcadei/eci_tok/round3
ls $E/tok_r3 > $E/r3/deleted_tok_r3_files.txt
ls $B > $E/r3/deleted_beegfs_adapt_files.txt
rm -rf $E/tok_r3
rm -f $B/*_adapt*
mkdir -p $B/codes $B/e4
cp $E/r3/codes/*.npy $B/codes/ && cmp <(cd $E/r3/codes && ls) <(cd $B/codes && ls) && rm -rf $E/r3/codes
for f in $E/r3/e4/*.npy $E/r3/e4/*.npz; do cp $f $B/e4/ && cmp $f $B/e4/$(basename $f) && rm $f; done
echo "kept on BeeGFS: $(ls $B/codes | wc -l) code files, $(ls $B/e4 | wc -l) e4 files"
ls $E/tok_r3 2>/dev/null && echo "tok_r3 STILL EXISTS" || echo "tok_r3 deleted"
