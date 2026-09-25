#!/usr/bin/env python
"""
Download every input a study site needs, in one unattended run.

Runs the independent download scripts of this folder one after the other,
each as its own subprocess:

    soilgrids   download_soilgrids.py          -> <root>/inputs/soilgrids/*.tif
    sigma0      download_sentinel1_gee.py      -> <root>/sigma0/*.nc
    ndvi        download_sentinel2_ndvi_gee.py -> <root>/ndvi/*.nc
    merida      download_merida.py             -> <root>/merida/*.nc

A step is skipped when all of its expected output files already exist (so a
re-run only does what's missing), and a failed step is retried: the
sub-scripts skip what they already finished, so a retry resumes rather than
restarts. A failure never stops the other steps. Everything each step prints
goes to ``<root>/logs/``, together with a summary; the exit code is non-zero
if any step failed.

Before downloading anything, a preflight check verifies the field
shapefiles, the credentials each requested step needs (``GEE_PROJECT`` for
sigma0/ndvi, ``MERIDA_USER``/``MERIDA_PASSWORD`` for merida) and ``curl``, so
the run doesn't die hours in for a missing variable.

Usage
-----
    export WCM_DATA_ROOT=... GEE_PROJECT=... MERIDA_USER=... MERIDA_PASSWORD=...
    nohup python download_all.py --config configuration_00_preprocessing_TEMPLATE.json \\
        --merida-config configuration_00_preprocessing_MERIDA_TEMPLATE.json &

    python download_all.py --config ... --merida-config ... --steps sigma0 ndvi   # a subset
    python download_all.py --config ... --merida-config ... --dry-run             # only report

Author:  Martina Natali <martinanatali@cnr.it> (GitHub: martina01natali)
Created: 2026-09-24
License: GPL-3.0 (see LICENSE)
AI assistance: developed with the help of Claude Code (Anthropic,
               Claude Opus 5.5) under the author's direction; see AI_DISCLOSURE.md
"""
import os
import sys
import glob
import shutil
import argparse
import subprocess
from datetime import datetime

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(HERE)
sys.path.insert(0, REPO_ROOT)

from wcm_swb.config import get_data_settings, substitute_keywords  # noqa: E402

STEPS = ['soilgrids', 'sigma0', 'ndvi', 'merida']
SCRIPTS = {
    'soilgrids': 'download_soilgrids.py',
    'sigma0': 'download_sentinel1_gee.py',
    'ndvi': 'download_sentinel2_ndvi_gee.py',
    'merida': 'download_merida.py',
}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--config', required=True, help='configuration_00_preprocessing.json (site, period, S1/S2, SoilGrids)')
    p.add_argument('--merida-config', default=None,
                   help='configuration_00_preprocessing_MERIDA.json (required for the merida step)')
    p.add_argument('--steps', nargs='+', choices=STEPS, default=STEPS, help='Steps to run (default: all)')
    p.add_argument('--field', default=None, help='Only this field (shapefile stem); not applied to merida')
    p.add_argument('--retries', type=int, default=2, help='Extra attempts for a failed step (default 2)')
    p.add_argument('--dry-run', action='store_true', help='Only report what is present/missing, download nothing')
    return p.parse_args()


# ----------------------------------------------------------------------------
# Expected outputs — the same paths the sub-scripts write
def _fields(data_settings, only_field=None):
    root = data_settings['paths']['root']
    file_shapes = os.path.join(root, data_settings['paths']['data_input']['file_shapes'])
    fields = [os.path.splitext(os.path.basename(f))[0] for f in sorted(glob.glob(file_shapes))]
    only_field = only_field or data_settings['options'].get('opt_field')
    return [f for f in fields if only_field is None or f == only_field], file_shapes


def _stem(data_settings, folder_key, filename_key, field):
    opt = data_settings['options']
    outputs_cfg = data_settings['paths']['data_output']
    filename = substitute_keywords(outputs_cfg[filename_key], opt_field=field,
                                   start_date=opt['start_date'], end_date=opt['end_date'])
    return os.path.join(data_settings['paths']['root'], outputs_cfg[folder_key], filename)


def expected_soilgrids(data_settings, fields):
    cfg = data_settings['soilgrids']
    return [f"{_stem(data_settings, 'folder_soil', 'filename_soil', field)}_{name}_{depth}.tif"
            for field in fields for name in cfg['properties'] for depth in cfg['depths']]


def expected_sigma0(data_settings, fields):
    return [_stem(data_settings, 'folder_sigma0', 'filename_sigma0', field) + '.nc' for field in fields]


def expected_ndvi(data_settings, fields):
    return [_stem(data_settings, 'folder_ndvi', 'filename_ndvi', field) + '.nc' for field in fields]


def expected_merida(merida_settings):
    """Merged per-variable files (``opt_merge``), else every monthly file."""
    opt = merida_settings['options']
    paths = merida_settings['paths']
    root = os.path.expandvars(paths['root'])
    out = paths['data_output']
    yyyymm = [f'{int(y):04d}{int(m):02d}' for y in sorted(opt['years']) for m in sorted(opt['months'])]
    if opt.get('opt_merge'):
        return [os.path.join(root, out['folder_merged'],
                             substitute_keywords(out['filename_merged'], label=label, start=yyyymm[0], end=yyyymm[-1]))
                for label in opt['variables'].values()]
    return [os.path.join(root, out['folder_monthly'], substitute_keywords(out['filename_monthly'], variable=v, yyyymm=m))
            for v in opt['variables'] for m in yyyymm]


# ----------------------------------------------------------------------------
def preflight(steps, data_settings, merida_settings, fields, file_shapes):
    """Problems that would make a requested step fail for sure."""
    problems = []
    if not fields:
        problems.append(f'no field shapefiles matching {file_shapes}')
    for step in steps:
        if not os.path.exists(os.path.join(HERE, SCRIPTS[step])):
            problems.append(f'{step}: {SCRIPTS[step]} not found in {HERE}')
    if {'sigma0', 'ndvi'} & set(steps):
        project = os.path.expandvars(data_settings['options']['gee_project'])
        if not project or project.startswith('$'):
            problems.append('sigma0/ndvi: options.gee_project is not set (export GEE_PROJECT=<your project id>)')
    if 'merida' in steps:
        if merida_settings is None:
            problems.append('merida: --merida-config not given')
        else:
            merida_root = os.path.normpath(os.path.expandvars(merida_settings['paths']['root']))
            if merida_root != os.path.normpath(data_settings['paths']['root']):
                problems.append(f'merida: paths.root of the MERIDA config ({merida_root}) is not the site '
                                f'({data_settings["paths"]["root"]}); set both configs to the same site')
            server = merida_settings['server']
            for var in (server['user_env'], server['password_env']):
                if not os.environ.get(var):
                    problems.append(f'merida: environment variable {var} is not set')
            if shutil.which('curl') is None:
                problems.append('merida: curl not found on PATH')
    return problems


def acquire_lock(log_dir):
    """Create ``<log_dir>/download_all.lock`` holding our pid; None if another
    launcher is running on this site. A lock left by a dead process is replaced."""
    lock = os.path.join(log_dir, 'download_all.lock')
    for _ in range(2):
        try:
            fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            try:
                pid = int(open(lock).read().strip() or 0)
                os.kill(pid, 0)
                return None  # alive: another launcher owns this site
            except (ValueError, ProcessLookupError):
                os.remove(lock)  # stale
                continue
            except PermissionError:
                return None
        with os.fdopen(fd, 'w') as fh:
            fh.write(str(os.getpid()))
        return lock
    return None


def run_step(step, cmd, log_file, retries):
    """Run one sub-script (appending its output to `log_file`); True on success."""
    env = dict(os.environ)
    env['PYTHONPATH'] = REPO_ROOT + os.pathsep + env.get('PYTHONPATH', '')
    for attempt in range(1, retries + 2):
        with open(log_file, 'a') as log:
            log.write(f'\n===== {datetime.now():%Y-%m-%d %H:%M:%S} {step}, attempt {attempt}: {" ".join(cmd)}\n')
            log.flush()
            result = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                                    cwd=os.path.dirname(log_file), env=env)
        if result.returncode == 0:
            return True
        print(f'  {step}: attempt {attempt} failed (exit {result.returncode}), see {log_file}')
    return False


def main():
    args = parse_args()
    config = os.path.abspath(args.config)
    data_settings = get_data_settings(config)
    data_settings['paths']['root'] = root = os.path.expandvars(data_settings['paths']['root'])
    merida_config = os.path.abspath(args.merida_config) if args.merida_config else None
    merida_settings = get_data_settings(merida_config) if merida_config else None
    steps = [s for s in STEPS if s in args.steps]
    fields, file_shapes = _fields(data_settings, args.field)

    expected = {
        'soilgrids': lambda: expected_soilgrids(data_settings, fields),
        'sigma0': lambda: expected_sigma0(data_settings, fields),
        'ndvi': lambda: expected_ndvi(data_settings, fields),
        'merida': lambda: expected_merida(merida_settings) if merida_settings else [],
    }
    print(f'Site: {root}\nFields: {", ".join(fields) or "-"}\nSteps: {", ".join(steps)}')
    if 'merida' in steps and merida_settings:
        years = {int(y) for y in merida_settings['options']['years']}
        opt = data_settings['options']
        needed = set(range(int(opt['start_date'][:4]), int(opt['end_date'][:4]) + 1))
        if needed - years:
            print(f'  WARNING: MERIDA years {sorted(years)} do not cover the Sentinel period '
                  f'{opt["start_date"]}..{opt["end_date"]} (missing {sorted(needed - years)}); '
                  f'set options.years in {os.path.basename(merida_config)}')

    todo = []
    for step in steps:
        files = expected[step]()
        missing = [f for f in files if not os.path.exists(f)]
        status = 'present, skip' if files and not missing else f'{len(missing)}/{len(files)} file(s) missing'
        print(f'  {step:10s} {status}')
        if missing or not files:
            todo.append(step)
    if args.dry_run or not todo:
        print('Nothing to download.' if not todo else 'Dry run: nothing downloaded.')
        return 0

    problems = preflight(todo, data_settings, merida_settings, fields, file_shapes)
    if problems:
        print('Preflight failed, nothing downloaded:\n  - ' + '\n  - '.join(problems))
        return 2

    log_dir = os.path.join(root, 'logs')
    os.makedirs(log_dir, exist_ok=True)
    lock = acquire_lock(log_dir)
    if lock is None:
        print(f'Another download_all.py is running on this site ({log_dir}/download_all.lock); not starting.')
        return 3
    try:
        return _run(args, steps, todo, expected, config, merida_config, log_dir)
    finally:
        os.remove(lock)


def _run(args, steps, todo, expected, config, merida_config, log_dir):
    stamp = f'{datetime.now():%Y%m%d-%H%M%S}'
    results = {}
    for step in todo:
        cmd = [sys.executable, '-u', os.path.join(HERE, SCRIPTS[step]),
               '--config', merida_config if step == 'merida' else config]
        if args.field and step != 'merida':
            cmd += ['--field', args.field]
        log_file = os.path.join(log_dir, f'download_{step}_{stamp}.log')
        print(f'{datetime.now():%H:%M:%S} {step}: running (log: {log_file})')
        ok = run_step(step, cmd, log_file, args.retries)
        still_missing = [f for f in expected[step]() if not os.path.exists(f)]
        results[step] = 'OK' if ok and not still_missing else (
            f'FAILED' if not ok else f'finished but {len(still_missing)} expected file(s) missing')
        print(f'{datetime.now():%H:%M:%S} {step}: {results[step]}')

    summary = os.path.join(log_dir, f'download_all_{stamp}.log')
    with open(summary, 'w') as fh:
        for step in steps:
            fh.write(f'{step}: {results.get(step, "skipped (outputs present)")}\n')
    print(f'Summary written to {summary}')
    return 0 if all(r == 'OK' for r in results.values()) else 1


if __name__ == '__main__':
    sys.exit(main())
