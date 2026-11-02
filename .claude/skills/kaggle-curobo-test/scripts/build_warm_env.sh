#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Build the warm cuRobo test environment on a Kaggle GPU kernel.
#
# Run as the --cmd of a kaggle_run.sh run with --python and slug kgt-env-<name>. It runs the
# full suite once, which compiles every Warp kernel into the cache, then packs the venv, its
# interpreter and the compile caches into /kaggle/working/env.tar. A later run with
# --env <name> mounts that output and unpacks it to /, restoring all of them.

bash "$(dirname "$0")/run_split_suite.sh"
echo "SUITE_RC $?"

WARP_CACHE=$(python -c "import warp as wp; wp.init(); print(wp.config.kernel_cache_dir)" 2>/dev/null | tail -1)
echo "WARP_CACHE $WARP_CACHE"
du -sh "$HOME"/.cache/* "$WARP_CACHE" 2>/dev/null | sed 's/^/CACHE_DU /'

# Paths are relative to /, where --env unpacks the archive.
paths="tmp/kgt-venv tmp/kgt-uv-python ${WARP_CACHE#/}"
[ -d "$HOME/.cache/torch_extensions" ] && paths="$paths ${HOME#/}/.cache/torch_extensions"
tar -cf /kaggle/working/env.tar -C / $paths
ls -la /kaggle/working/env.tar | sed 's/^/ENV_TAR /'
