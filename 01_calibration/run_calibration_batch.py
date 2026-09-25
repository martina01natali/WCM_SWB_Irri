#!/usr/bin/env python
"""
Multi-field calibration driver: launches ``run_calibration.py`` in a fresh
subprocess per field found by the config's ``file_shapes`` glob.

Subprocess-per-field isolates PyMC/numba compiled-graph state between
fields.

Author:  Martina Natali <martinanatali@cnr.it> (GitHub: martina01natali)
Created: 2026-09-23
License: GPL-3.0 (see LICENSE)
AI assistance: developed with the help of Claude Code (Anthropic,
               Claude Opus 5.5) under the author's direction; see AI_DISCLOSURE.md

Usage
-----
    python run_calibration_batch.py --config configuration_01_calibration_TEMPLATE.json
"""

import os
import sys
import glob
import time
import logging
import argparse
import subprocess

from wcm_swb.config import get_data_settings


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', required=True, help='Path to configuration_01_calibration.json')
    return p.parse_args()


def main():
    args = parse_args()
    data_settings = get_data_settings(args.config)
    root = os.path.expandvars(data_settings['paths']['root'])
    file_shapes = os.path.join(root, data_settings['paths']['data_input']['file_shapes'])
    fields = sorted(f.split('/')[-1].split('.')[0] for f in glob.glob(file_shapes))

    logging.basicConfig(filename='main_01_calibration.log', level=logging.INFO,
                         format='%(asctime)s %(levelname)-8s %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    logging.info('Found %d field(s): %s', len(fields), fields)

    for field in fields:
        start = time.time()
        logging.info('Starting field: %s', field)
        result = subprocess.run(
            [sys.executable, os.path.join(os.path.dirname(__file__), 'run_calibration.py'),
             '--config', args.config, '--field', field],
            check=False, capture_output=True, text=True,
        )
        elapsed = round((time.time() - start) / 60, 2)
        if result.returncode != 0:
            logging.error('Field %s FAILED after %s min:\n%s', field, elapsed, result.stderr[-4000:])
        else:
            logging.info('Field %s done in %s min.', field, elapsed)


if __name__ == '__main__':
    main()
