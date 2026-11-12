#!/usr/bin/env python3
"""Clone immutable upstream revisions; never overwrite an existing checkout."""
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parent


def setup():
    (ROOT / 'upstream').mkdir(exist_ok=True)
    for name, spec in json.loads((ROOT / 'upstreams.json').read_text()).items():
        target = ROOT / 'upstream' / name
        if target.exists():
            actual = subprocess.check_output(['git', '-C', str(target), 'rev-parse', 'HEAD'], text=True).strip()
            dirty = subprocess.check_output(['git', '-C', str(target), 'status', '--porcelain'], text=True)
            if actual != spec['revision'] or dirty:
                raise SystemExit(f'{target}: existing checkout differs from lock or has local changes; left untouched')
        else:
            subprocess.run(['git', 'clone', '--no-checkout', spec['url'], str(target)], check=True)
            subprocess.run(['git', '-C', str(target), 'checkout', '--detach', spec['revision']], check=True)
        print(f'{name}: {spec["revision"]}', flush=True)


if __name__ == '__main__':
    setup()
