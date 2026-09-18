#!/usr/bin/env python3
"""Reject infrastructure-specific files, machine paths and credential material."""
from pathlib import Path
import argparse
import os
import re
import subprocess
import sys
import zipfile
from urllib.parse import unquote, urlsplit

ROOT = Path(__file__).resolve().parents[1]
PATTERNS = {
    'cluster integration': re.compile(r'os' + r'mo|swift' + r'://|pdx[.]s8k|AUTH_' + r'team|isaac-' + r'dev-|am' + r'lfs-', re.I),
    'workstation path': re.compile(r'/ho' + r'me/[^/\s]+|/mnt/' + r'amlfs'),
    'credential material': re.compile(r'gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{40,}|hf_[A-Za-z0-9]{25,}|sk-[A-Za-z0-9]{25,}|BEGIN (?:RSA |OPENSSH )?PRIVATE' + r' KEY'),
    'hardware serial': re.compile(r'(?:Serial_|so101_(?:leader_)?)[A-Fa-f0-9]{8,}(?![A-Fa-f0-9])|(?<![A-Za-z0-9])(?=[A-F0-9]{0,9}[A-F])[A-F0-9]{10}(?![A-Za-z0-9])'),
    'obsolete repository': re.compile(r'github[.]com/' + r'es-rl/so101' + r'_keyboard_task'),
}


def source_files():
    if (ROOT / '.git').exists():
        names = subprocess.check_output(
            ['git', 'ls-files', '--cached', '--others', '--exclude-standard', '-z'],
            cwd=ROOT).decode().split('\0')
        return [(ROOT / name, name) for name in sorted(set(names)) if name]
    # Source archives must not accidentally query a parent checkout or scan run outputs.
    ignored = {'.git', '.artifacts', 'output', 'logs', 'local', '.venv-hardware', '__pycache__', '.pytest_cache', 'build', 'dist'}
    files = []
    for directory, subdirectories, names in os.walk(ROOT):
        subdirectories[:] = [name for name in subdirectories if name not in ignored and not name.endswith('.egg-info')]
        for name in names:
            path = Path(directory) / name
            files.append((path, str(path.relative_to(ROOT))))
    return files


def source_link_failures(files):
    """Require local Markdown links to work from the distributed source alone."""
    included = {path.resolve() for path, _ in files if path.is_file()}
    failures = []
    for path, relative in files:
        if path.suffix != '.md' or not path.is_file():
            continue
        contents = path.read_text()
        for match in re.finditer(r'\[[^\]\n]*\]\(([^)\n]+)\)', contents):
            link = match.group(1).split(' "', 1)[0].strip('<>')
            url = urlsplit(link)
            if url.scheme or url.netloc or not url.path:
                continue
            target = (path.parent / unquote(url.path)).resolve()
            if target not in included and not (target.is_dir() and any(p.is_relative_to(target) for p in included)):
                line = contents.count('\n', 0, match.start()) + 1
                failures.append(f'{relative}:{line}: local link not included in source')
    return failures


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--artifact-dir', type=Path, help='also scan public bundle metadata and checkpoint pickle strings')
    args = parser.parse_args()
    files = source_files()
    failures = source_link_failures(files)
    if args.artifact_dir:
        if not args.artifact_dir.is_dir():
            parser.error('artifact directory does not exist')
        files += [(path, 'artifacts/' + str(path.relative_to(args.artifact_dir)))
                  for path in args.artifact_dir.rglob('*')
                  if path.is_file() and '.cache' not in path.relative_to(args.artifact_dir).parts]
    for path, relative in files:
        if not path.is_file():
            continue
        if path.suffix == '.pt' and zipfile.is_zipfile(path):
            with zipfile.ZipFile(path) as archive:
                payload = b'\n'.join(archive.read(name) for name in archive.namelist() if name.endswith('.pkl'))
        elif path.suffix in ('.mp4', '.png', '.npz'):
            continue
        else:
            payload = path.read_bytes()
        text = relative + '\n' + payload.decode(errors='replace')
        for label, pattern in PATTERNS.items():
            if pattern.search(text):
                failures.append(f'{relative}: {label}')
    if failures:
        # Report locations only; never echo matched credentials or internal URLs.
        print('\n'.join(failures), file=sys.stderr)
        return 1
    print('Public source check passed: tracked/new files are clean and local Markdown links resolve in the source bundle.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
