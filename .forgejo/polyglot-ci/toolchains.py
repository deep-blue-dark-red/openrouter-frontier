#!/usr/bin/env python3
"""Provision user-owned toolchains on the external CI volume, without sudo."""
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import urllib.request
import pipeline

def run(argv):
    print('+', ' '.join(argv), flush=True)
    logfile = Path.cwd() / '.polyglot-ci-work/toolchains.log'
    with logfile.open('a') as log:
        log.write('+ ' + ' '.join(argv) + '\n'); log.flush()
        result = subprocess.run(argv, stdout=log, stderr=subprocess.STDOUT, timeout=1800)
    if result.returncode:
        print(logfile.read_text(errors='replace')[-16000:], flush=True)
        raise RuntimeError('Toolchain preparation failed; see toolchains.log')

def expose(path):
    os.environ['PATH'] = str(path) + os.pathsep + os.environ['PATH']
    with open(os.environ['GITHUB_PATH'], 'a') as f: f.write(str(path) + '\n')

def main():
    root = Path.cwd()
    storage = Path('/mnt/commons/git-ci/polyglot')
    if not root.resolve().is_relative_to(storage.resolve()):
        raise RuntimeError('Runner workspace must be on /mnt/commons/git-ci/polyglot before running builds')
    tools = storage / 'toolchains'
    tools.mkdir(exist_ok=True)
    plan = json.loads((root / '.polyglot-ci-work/plan.json').read_text())
    kinds = {u['kind'] for u in plan['projects']}
    config = plan['config'].get('toolchains', {})
    if kinds & {'cmake', 'meson', 'python'}:
        helpers = tools / 'python-build'
        if not (helpers/'bin/python').exists(): run(['python3','-m','venv',str(helpers)])
        if not (helpers/'bin/cmake').exists() or not (helpers/'bin/meson').exists():
            run([str(helpers/'bin/python'),'-m','pip','install','cmake==4.1.3','meson==1.9.2','ninja==1.13.0'])
        expose(helpers/'bin')
    if 'dotnet' in kinds:
        dotnet = tools / 'dotnet'
        dotnet.mkdir(exist_ok=True)
        installer = tools/'dotnet-install.sh'
        if not installer.exists():
            with urllib.request.urlopen('https://dot.net/v1/dotnet-install.sh',timeout=60) as response:
                installer.write_bytes(response.read())
        files = pipeline.tracked(root)
        pins = set(config.get('dotnet', []))
        majors = set()
        for p in files:
            if any(part in pipeline.SKIP for part in p.parts[:-1]): continue
            if p.name == 'global.json':
                sdk = json.loads((root/p).read_text()).get('sdk',{})
                version = sdk.get('version')
                if version:
                    if sdk.get('rollForward', 'patch') in ('latestFeature','feature','latestMinor','minor','latestMajor','major'):
                        majors.add(version.split('.')[0])
                    else: pins.add(version)
            elif p.suffix in ('.csproj','.fsproj','.vbproj','.props'):
                majors.update(re.findall(r'\bnet(\d+)\.\d+',pipeline.text(root/p)))
        if not pins and not majors: majors.add('10')
        installed = {p.name for p in (dotnet/'sdk').iterdir()} if (dotnet/'sdk').exists() else set()
        for version in sorted(pins):
            if version not in installed:
                run(['bash',str(installer),'--version',version,'--install-dir',str(dotnet),'--no-path'])
        installed = {p.name for p in (dotnet/'sdk').iterdir()} if (dotnet/'sdk').exists() else set()
        for major in sorted(majors, key=int):
            if int(major)<6: continue
            if not any(v.startswith(major+'.') for v in installed):
                run(['bash',str(installer),'--channel',major+'.0','--quality','preview' if int(major)>=11 else 'GA',
                     '--install-dir',str(dotnet),'--no-path'])
        expose(dotnet)
        with open(os.environ['GITHUB_ENV'],'a') as f: f.write('DOTNET_ROOT='+str(dotnet)+'\n')
        run([str(dotnet/'dotnet'),'--list-sdks'])
    if 'zig' in kinds:
        # Zig's minimum version is not an exact pin; do not silently downgrade it.
        wanted = config.get('zig')
        actual = subprocess.check_output(['zig','version'],text=True).strip() if shutil.which('zig') else None
        if wanted and wanted != actual:
            venv = tools / ('zig-'+wanted)
            if not (venv/'bin/zig').exists():
                run(['python3','-m','venv',str(venv)])
                run([str(venv/'bin/python'),'-m','pip','install','ziglang=='+wanted])
            expose(venv/'bin')
        run(['zig','version'])
    if 'rust' in kinds:
        if Path('/usr/local/cargo/bin').exists(): expose(Path('/usr/local/cargo/bin'))
        run(['cargo','--version'])
    if 'go' in kinds: run(['go','version'])
    if 'node' in kinds: run(['node','--version'])

if __name__=='__main__': main()
