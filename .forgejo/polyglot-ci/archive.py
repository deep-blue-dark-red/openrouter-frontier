#!/usr/bin/env python3
"""Copy CI reports onto shared storage without copying ownership or ACLs."""
import json
import os
from pathlib import Path
import shutil

def copy_reports(source, destination):
    destination.mkdir(parents=True, exist_ok=True)
    count = 0
    candidates = [p for p in source.iterdir() if p.is_file()]
    for name in ('tests', 'benchmarks'):
        directory = source / name
        if directory.is_dir(): candidates.extend(p for p in directory.rglob('*') if p.is_file())
    for path in candidates:
        # Artifacts must not pull in arbitrary files through links created by a build.
        if path.is_symlink() or not path.resolve().is_relative_to(source.resolve()): continue
        target = destination / path.relative_to(source)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, target)
        count += 1
    return count

def main():
    source = Path('.polyglot-ci-work')
    if not source.exists(): return
    base = Path('/mnt/commons/git-ci/polyglot/results')
    destination = base / os.environ['GITHUB_REPOSITORY'] / os.environ['GITHUB_RUN_ID'] / os.environ.get('GITHUB_RUN_ATTEMPT', '1')
    if not destination.resolve().is_relative_to(base.resolve()): raise ValueError('Invalid result path')
    outcome = {}
    try:
        outcome['files'] = copy_reports(source, destination)
        outcome['status'] = 'passed'
        report = json.loads((source/'report.json').read_text()) if (source/'report.json').exists() else {}
        summary = f"Stages: {report.get('stages', {})}\nReports: {destination}\n"
        print(summary)
        if os.environ.get('GITHUB_STEP_SUMMARY'):
            with open(os.environ['GITHUB_STEP_SUMMARY'], 'a') as f: f.write(summary)
    except Exception as error:
        outcome.update(status='failed', error=str(error))
        raise
    finally:
        destination.mkdir(parents=True, exist_ok=True)
        (destination/'archive.json').write_text(json.dumps(outcome, indent=2))

if __name__=='__main__': main()
