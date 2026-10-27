#!/usr/bin/env bash
set -e
python src/00_download.py      # about 130 MB, 2.1M rows (skip if data/sparcs_2022_los.csv.gz exists)
python src/01_load_validate.py
python src/02_eda.py
python src/03_models.py
