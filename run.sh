#!/bin/bash
set -euxo
cd "$(dirname "$0")"
source /opt/venv/bin/activate
pip install -e moshi/.
SSL_DIR=$(mktemp -d)
python -m moshi.server --fp8 --ssl "$SSL_DIR"
