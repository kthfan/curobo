#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Run the full cuRobo test suite on a Kaggle kernel with two GPUs, one half per GPU.
#
# Usage (from the repository root, as the kaggle_run.sh --cmd):
#   bash .claude/skills/kaggle-curobo-test/scripts/run_split_suite.sh [extra pytest args]
#
# Group A (GPU 0) holds the tests whose kernels take longest to compile or run; group B (GPU 1)
# holds the rest. With the warm environment both finish in about 8-10 minutes on 2x T4.
# Prints A_DONE/B_DONE/SUITE_A/SUITE_B/TOTAL lines and the failures, and exits non-zero when
# either half fails.

GROUP_A="curobo/tests/_src/perception curobo/tests/_src/geom curobo/tests/_src/motion"
IGNORE_A=""
for path in $GROUP_A; do IGNORE_A="$IGNORE_A --ignore=$path"; done
ARGS="-n 0 -p no:cacheprovider -q -rfE $*"

start=$(date +%s)
(
    CUDA_VISIBLE_DEVICES=0 python -m pytest $ARGS $GROUP_A > /tmp/suite_a.log 2>&1
    echo $? > /tmp/suite_a.rc
    echo "A_DONE $(( $(date +%s) - start ))s"
) &
(
    CUDA_VISIBLE_DEVICES=1 python -m pytest $ARGS curobo/tests $IGNORE_A > /tmp/suite_b.log 2>&1
    echo $? > /tmp/suite_b.rc
    echo "B_DONE $(( $(date +%s) - start ))s"
) &
wait

echo "SUITE_A $(tail -1 /tmp/suite_a.log)"
echo "SUITE_B $(tail -1 /tmp/suite_b.log)"
echo "TOTAL $(( $(date +%s) - start ))s"
grep -h "^FAILED\|^ERROR" /tmp/suite_a.log /tmp/suite_b.log | head -50
cp /tmp/suite_a.log /tmp/suite_b.log /kaggle/working/ 2>/dev/null
[ "$(cat /tmp/suite_a.rc)" = 0 ] && [ "$(cat /tmp/suite_b.rc)" = 0 ]
