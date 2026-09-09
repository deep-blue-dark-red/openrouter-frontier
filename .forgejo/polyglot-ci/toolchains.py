#!/usr/bin/env python3
"""Provision user-owned toolchains on the external CI volume, without sudo."""
import json
import os
from pathlib import Path
import platform
import re
import shutil
import subprocess
import sys
import tarfile
import urllib.request
import pipeline

def run(argv):
    print('+', ' '.join(argv), flush=True)
    logfile = Path.cwd() / '.polyglot-ci-work/toolchains.log'
    with logfile.open('a') as log:
        log.write('+ ' + ' '.join(argv) + '\n'); log.flush()
        try: result = subprocess.run(argv, stdout=log, stderr=subprocess.STDOUT, timeout=1800)
        except subprocess.TimeoutExpired:
            log.write('\nToolchain command timed out.\n')
            raise RuntimeError('Toolchain command timed out: ' + ' '.join(argv)) from None
        except OSError as error:
            log.write(str(error) + '\n')
            raise RuntimeError(f'Cannot execute {argv[0]}: {error}') from error
    if result.returncode:
        print(logfile.read_text(errors='replace')[-16000:], flush=True)
        raise RuntimeError('Toolchain preparation failed; see toolchains.log')

def expose(path):
    os.environ['PATH'] = str(path) + os.pathsep + os.environ['PATH']
    with open(os.environ['GITHUB_PATH'], 'a') as f: f.write(str(path) + '\n')

def machine_tag(machine=None):
    machine = machine or platform.machine()
    if machine in ('x86_64', 'amd64'): return {'node': 'x64', 'go': 'amd64'}
    if machine in ('aarch64', 'arm64'): return {'node': 'arm64', 'go': 'arm64'}
    raise RuntimeError('Unsupported CPU architecture: ' + machine)

def fetch(url, target):
    with urllib.request.urlopen(url, timeout=300) as response, target.open('wb') as file:
        shutil.copyfileobj(response, file)

def unpack(archive, destination):
    with tarfile.open(archive) as bundle:
        try: bundle.extractall(destination, filter='data')
        except TypeError: bundle.extractall(destination)

def node_index():
    with urllib.request.urlopen('https://nodejs.org/dist/index.json', timeout=60) as response:
        return json.load(response)

def resolve_node(wanted, entries=None):
    entries = entries if entries is not None else node_index()
    if wanted:
        for entry in entries:
            if entry['version'] == 'v' + wanted or entry['version'].startswith('v' + wanted + '.'): return entry['version'][1:]
        raise RuntimeError('No Node.js release matches ' + wanted)
    return next(entry['version'][1:] for entry in entries if entry.get('lts'))

def ensure_node(tools, config):
    if shutil.which('npm'): return run(['npm', '--version'])
    version = resolve_node(config.get('node'))
    directory = tools / ('node-v' + version)
    if not (directory / 'bin/npm').exists():
        tag = machine_tag()['node']
        archive = tools / 'node.tar.gz'
        fetch(f'https://nodejs.org/dist/v{version}/node-v{version}-linux-{tag}.tar.gz', archive)
        staging = tools / ('.node-stage-' + version)
        shutil.rmtree(staging, ignore_errors=True); staging.mkdir(parents=True)
        unpack(archive, staging)
        extracted = [child for child in staging.iterdir() if child.is_dir()]
        if len(extracted) != 1: raise RuntimeError('Unexpected Node.js archive layout')
        shutil.rmtree(directory, ignore_errors=True)
        extracted[0].replace(directory)
        shutil.rmtree(staging); archive.unlink()
    expose(directory / 'bin')
    run([str(directory / 'bin/node'), '--version'])
    run([str(directory / 'bin/npm'), '--version'])

def go_index():
    with urllib.request.urlopen('https://go.dev/dl/?mode=json', timeout=60) as response:
        return json.load(response)

def resolve_go(wanted, entries=None):
    if wanted: return wanted.lstrip('go')
    entries = entries if entries is not None else go_index()
    return entries[0]['version'].lstrip('go')

def ensure_go(tools, config):
    if shutil.which('go'): return run(['go', 'version'])
    version = resolve_go(config.get('go'))
    directory = tools / ('go-' + version)
    if not (directory / 'bin/go').exists():
        tag = machine_tag()['go']
        archive = tools / 'go.tar.gz'
        fetch(f'https://go.dev/dl/go{version}.linux-{tag}.tar.gz', archive)
        staging = tools / ('.go-stage-' + version)
        shutil.rmtree(staging, ignore_errors=True); staging.mkdir(parents=True)
        unpack(archive, staging)
        shutil.rmtree(directory, ignore_errors=True)
        (staging / 'go').replace(directory)
        shutil.rmtree(staging); archive.unlink()
    expose(directory / 'bin')
    run([str(directory / 'bin/go'), 'version'])

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
    if 'go' in kinds: ensure_go(tools, config)
    if 'node' in kinds: ensure_node(tools, config)

if __name__=='__main__': main()
