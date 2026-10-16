#!/bin/bash
#
# Sweeper for scripts/eci/atlas_v37_methods.sh: waits (sleeping inside this 1-CPU job, never polling from the login
# node) for each R4 d2 fnorm NES result set of the rollout (scripts/eci/rollout_r4_nes.sh) and submits the methods job
# on it once. A result set is ready when the SUMMARY.md of its three outcome runs exists (root, maxpool_bouts/,
# maxpool_latency/; ants also the same three under pairs/, which the page reads) AND either the rollout's last step
# has written its file (ants: pairs_noday/maxpool_bouts/SUMMARY.md, mice: period_adjust/SUMMARY.md, frogs: the latency
# SUMMARY.md) or none of those files changed for 15 min. Targets: ants_v3 (eci_r4v3), mice, frogs, ants_v4 (eci_r4v4,
# when its best.json and result set appear). SAE names from results/vision/eci_rollout_r4/<dom>/sae/best.json 'd2'.
#   sbatch scripts/eci/atlas_v37_methods_sweep.sh
#SBATCH --job-name=v37_sweep
#SBATCH --output=results/vision/eci_atlas_v37/methods/logs/%x_%j.out
#SBATCH --time=12:00:00
#SBATCH --partition=defaultp
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
set -uo pipefail
REPO=/nfs/scistore19/locatgrp/rcadei/artificial-causal-inference
cd $REPO
R=results/vision
declare -A ROOTS=([ants_v3]=$R/ants/eci_r4v3/nes [mice]=$R/mice/eci/nes [frogs]=$R/frogs/eci/nes [ants_v4]=$R/ants/eci_r4v4/nes)
declare -A DONE=()
best() {  # d2 SAE name of a rollout domain, or empty
    local f=$R/eci_rollout_r4/$1/sae/best.json
    [ -f $f ] && python3 -c "import json,sys; print(json.load(open('$f'))['d2']['name'])" 2>/dev/null
}
ready() {  # $1 domain, $2 result set
    local d=$1 res=$2 files=() last
    files=($res/SUMMARY.md $res/maxpool_bouts/SUMMARY.md $res/maxpool_latency/SUMMARY.md)
    case $d in
        ants_*) files+=($res/pairs/SUMMARY.md $res/pairs/maxpool_bouts/SUMMARY.md $res/pairs/maxpool_latency/SUMMARY.md)
                last=$res/pairs_noday/maxpool_bouts/SUMMARY.md ;;
        mice) last=$res/period_adjust/SUMMARY.md ;;
        *) last=$res/maxpool_latency/SUMMARY.md ;;
    esac
    for f in "${files[@]}"; do [ -f $f ] || return 1; done
    [ -f $last ] && return 0
    local newest=$(stat -c %Y "${files[@]}" | sort -n | tail -1)
    [ $(( $(date +%s) - newest )) -gt 900 ] && return 0
    return 1
}
while true; do
    left=0
    for d in ants_v3 mice frogs ants_v4; do
        [ -n "${DONE[$d]:-}" ] && continue
        left=1
        name=$(best $d)
        [ -z "$name" ] && continue
        res=${ROOTS[$d]}/${name}_fnorm
        [ -d $res ] || continue
        if ready $d $res; then
            jid=$(RES=$res sbatch --parsable --export=ALL --job-name=v37_${d} scripts/eci/atlas_v37_methods.sh)
            echo "$(date -Is) $d: $res ready -> submitted job $jid"
            DONE[$d]=$jid
        fi
    done
    [ $left = 0 ] && { echo "$(date -Is) all submitted"; break; }
    sleep 300
done
