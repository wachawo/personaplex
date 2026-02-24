#!/bin/bash
set -euxo
cd "$(dirname "$0")"
if [ -z $VIRTUAL_ENV ]; then
  if [ ! -d .venv ]; then
    python3 -m venv .venv
  fi
  source .venv/bin/activate
  pip3 install -r moshi/requirements.txt
fi
pip install -e moshi/.
SSL_DIR=$(mktemp -d)
python -m moshi.server --fp8 --ssl "$SSL_DIR"
