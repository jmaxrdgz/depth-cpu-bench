#!/usr/bin/env python3
"""Create the requested PRIVATE personal GitHub repository and push this checkout."""
import argparse
from pathlib import Path
import shutil
import subprocess


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--name', default='depth-cpu-bench')
    args = parser.parse_args()
    gh = shutil.which('gh')
    if gh is None:
        raise SystemExit('Install GitHub CLI and run gh auth login first.')
    root = Path(__file__).resolve().parent
    subprocess.run([gh, 'auth', 'status'], check=True)
    subprocess.run([gh, 'repo', 'create', args.name, '--private', '--source', str(root),
                    '--remote', 'origin', '--push', '--description',
                    'CPU latency benchmarks for FastDepth, GuideDepth and RT-MonoDepth using synthetic noise'], check=True)
    owner = subprocess.check_output([gh, 'api', 'user', '--jq', '.login'], text=True).strip()
    privacy = subprocess.check_output([gh, 'repo', 'view', f'{owner}/{args.name}',
                                       '--json', 'isPrivate', '--jq', '.isPrivate'], text=True).strip()
    if privacy != 'true':
        raise SystemExit('Repository privacy verification failed')
    print(f'Confirmed private: https://github.com/{owner}/{args.name}')


if __name__ == '__main__':
    main()
