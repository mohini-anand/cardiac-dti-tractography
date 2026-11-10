#!/usr/bin/env bash
# Run the whole pipeline from the repository root: raw DICOM to the cleaned tractogram.
# Usage: bash run.sh [config]; the config defaults to config.yaml. The work folder named in the config (paths.work) is
# deleted at the start, so no file from an earlier run is reused, and again at the end unless KEEP_WORK=1.
set -euo pipefail

CONFIG="${1:-config.yaml}"
WORK=$(python -c "import sys, yaml; print(yaml.safe_load(open(sys.argv[1], encoding='ascii'))['paths']['work'] or '')" "$CONFIG")
rm -rf -- "${WORK:?paths.work is empty in the config}"

python -m src.convert "$CONFIG"
python -m src.register "$CONFIG"
python -m src.denoise "$CONFIG"
python -m src.tensor "$CONFIG"
python -m src.mask "$CONFIG"
python -m src.fill "$CONFIG"
python -m src.track "$CONFIG"
python -m src.clean "$CONFIG"

if [ "${KEEP_WORK:-0}" != "1" ]; then
    rm -rf -- "${WORK:?}"
fi
