#!/usr/bin/env python3
"""Forgejo polyglot CI. Standard library only; see README for the contract."""
from __future__ import annotations
import argparse
import fnmatch
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import shlex
import signal
import subprocess
import sys
import time
import tomllib
import xml.etree.ElementTree as ET

VERSION = 1
SKIP = {'.git', '.github', '.forgejo', '.agents', '.claude', '.venv', 'venv',
        'node_modules', 'vendor', 'third_party', 'third-party', 'external',
        'fixtures', 'testdata', 'obj', 'bin', 'dist', 'build', 'target',
        'generated', '.polyglot-ci-work', 'site-packages', '__pycache__'}
STAGES = ('build', 'test', 'benchmark')

def read_json(path, default=None):
    return json.loads(path.read_text()) if path.exists() else default

def text(path):
    return path.read_text(errors='replace') if path.is_file() else ''

def under(path, parent):
    return path == parent or parent in path.parents

def tracked(root):
    result = subprocess.run(['git', 'ls-files', '-z'], cwd=root, check=True, capture_output=True)
    return [Path(os.fsdecode(p)) for p in result.stdout.split(b'\0') if p]

def command(argv, cwd='.', **kw):
    return dict(argv=argv, cwd=str(cwd), **kw)

def project(kind, root, **kw):
    return dict(kind=kind, root=str(root), build=[], test=[], benchmark=[], **kw)

def minimal_roots(paths):
    selected = []
    for p in sorted(set(paths), key=lambda p: (len(p.parts), str(p))):
        if not any(under(p, parent) for parent in selected): selected.append(p)
    return selected

def discover(root):
    cfg = read_json(root / '.forgejo/polyglot-ci.json', {})
    if cfg.get('version', VERSION) != VERSION: raise ValueError('Unsupported ci.json version')
    files = [p for p in tracked(root) if not any(part in SKIP or part.startswith('OLD_') for part in p.parts[:-1])
             and not any(fnmatch.fnmatch(str(p), pat) for pat in cfg.get('exclude', []))]
    paths = set(files)
    units = []
    if 'projects' in cfg:
        for spec in cfg['projects']:
            u = project(spec['kind'], Path(spec.get('root', '.')))
            u.update(spec)
            units.append(u)
    else:
        solutions = [p for p in files if p.suffix in ('.sln', '.slnx')]
        # Prefer slnx when a migrated repository retains both formats.
        solutions = [p for p in solutions if p.suffix == '.slnx' or p.with_suffix('.slnx') not in paths]
        solutions = [p for p in solutions if not any(q.parent != p.parent and under(p.parent, q.parent) for q in solutions)]
        projects = [p for p in files if p.suffix in ('.csproj', '.fsproj', '.vbproj')]
        if solutions or projects:
            units.append(project('dotnet', '.', targets=[str(p) for p in solutions or projects],
                                 projects=[str(p) for p in projects]))
        for kind, name in [('zig', 'build.zig'), ('cmake', 'CMakeLists.txt'), ('meson', 'meson.build'),
                           ('rust', 'Cargo.toml'), ('go', 'go.mod')]:
            for directory in minimal_roots([p.parent for p in files if p.name == name]):
                units.append(project(kind, directory))
        native_roots = [Path(u['root']) for u in units if u['kind'] in ('cmake', 'meson', 'zig')]
        for directory in minimal_roots([p.parent for p in files if p.name in ('Makefile', 'makefile', 'GNUmakefile')]):
            if not any(under(directory, r) for r in native_roots): units.append(project('make', directory))
        py_roots = minimal_roots([p.parent for p in files if p.name in ('pyproject.toml', 'setup.py', 'requirements.txt')])
        loose_py = [p for p in files if p.suffix == '.py' and not any(under(p, r) for r in py_roots)]
        if loose_py: py_roots.append(Path('.'))
        for directory in sorted(set(py_roots)):
            units.append(project('python', directory, files=[str(p) for p in files if p.suffix == '.py' and under(p, directory)
                and not any(other != directory and under(other, directory) and under(p, other) for other in py_roots)]))
        for directory in minimal_roots([p.parent for p in files if p.name == 'package.json']):
            units.append(project('node', directory))
    for i, u in enumerate(units):
        u['id'] = f'{i:02d}-{u["kind"]}'
        path = (root / u['root']).resolve()
        if not path.is_relative_to(root.resolve()): raise ValueError('Project root escapes repository')
        for stage in STAGES:
            u.setdefault(stage, [])
            for c in u[stage]:
                if not isinstance(c.get('argv'), list) or not all(isinstance(a, str) for a in c['argv']):
                    raise ValueError('Commands must use argv string arrays')
                c.setdefault('cwd', u['root'])
                if not (root / c['cwd']).resolve().is_relative_to(root.resolve()):
                    raise ValueError('Command cwd escapes repository')
    return dict(version=VERSION, projects=units, config=cfg, tracked_count=len(files))

class Runner:
    def __init__(self, root):
        self.root = root.resolve()
        self.work = self.root / '.polyglot-ci-work'
        self.work.mkdir(exist_ok=True)
        self.plan_path = self.work / 'plan.json'
        self.report_path = self.work / 'report.json'
        self.report = read_json(self.report_path, {'stages': {}, 'commands': [], 'machine': {
            'system': platform.platform(), 'machine': platform.machine(), 'cpu_count': os.cpu_count(),
            'commit': os.environ.get('GITHUB_SHA', ''), 'runner': os.environ.get('RUNNER_NAME', '')}})

    def save(self):
        self.report_path.write_text(json.dumps(self.report, indent=2))

    def run(self, c, stage, unit, capture=False):
        argv = c['argv']
        cwd = self.root / c.get('cwd', '.')
        env = os.environ.copy()
        env.update({'CI': 'true', 'DOTNET_CLI_TELEMETRY_OPTOUT': '1', 'DOTNET_NOLOGO': '1',
                    'CMAKE_BUILD_PARALLEL_LEVEL': '2', 'CARGO_BUILD_JOBS': '2', 'PYTHONUNBUFFERED': '1'})
        env.update(c.get('env', {}))
        log = self.work / f'{len(self.report["commands"]):04d}-{unit}-{stage}.log'
        print(f'[{stage}/{unit}] {shlex.join(argv)} (cwd={c.get("cwd", ".")})', flush=True)
        started = time.monotonic()
        rc = 127
        try:
            with log.open('w') as out:
                process = subprocess.Popen(argv, cwd=cwd, env=env, stdout=out, stderr=subprocess.STDOUT, start_new_session=True)
                try:
                    rc = process.wait(timeout=c.get('timeout', 1800 if stage != 'benchmark' else 3600))
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGTERM)
                    try: process.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        os.killpg(process.pid, signal.SIGKILL); process.wait()
                    rc = 124
                    out.write('\nCI command timed out; process group terminated.\n')
        except OSError as error:
            log.write_text(str(error))
        output = log.read_text(errors='replace')
        self.report['commands'].append(dict(stage=stage, unit=unit, argv=argv, cwd=str(cwd.relative_to(self.root)),
                                             exit_code=rc, seconds=time.monotonic()-started, log=log.name))
        self.save()
        if not capture or rc: print(output[-24000:], flush=True)
        if rc: raise RuntimeError(f'{stage}/{unit} failed with exit {rc}; see {log.name}')
        if c.get('require_tests'):
            reports = list((self.work / c['require_tests']).rglob('*.trx'))
            executed = 0
            for report in reports:
                for element in ET.parse(report).iter():
                    if element.tag.rsplit('}', 1)[-1] == 'Counters': executed += int(element.get('executed', '0'))
            if not executed: raise RuntimeError('Test runner succeeded without recording any executed tests')
        if c.get('require_benchmarks'):
            measured = False
            for report in (self.work / c['require_benchmarks']).rglob('*.json'):
                data = read_json(report, {})
                if isinstance(data, dict):
                    measured |= any(b.get('Statistics') and b['Statistics'].get('N', 0) > 0 for b in data.get('Benchmarks', []))
            if not measured: raise RuntimeError('BenchmarkDotNet produced no successful measurements')
        return output

    def cmd(self, argv, u, stage='build', capture=False, **kw):
        return self.run(command(argv, u['root'], **kw), stage, u['id'], capture)

    def configure(self, u):
        kind, directory, uid = u['kind'], self.root / u['root'], u['id']
        if kind == 'custom': return
        # Explicit commands replace an adapter, and permit bespoke test / benchmark protocols.
        if any(u[s] for s in STAGES): return
        if kind == 'dotnet':
            targets = u.get('targets', [])
            if not targets: raise ValueError('dotnet project requires targets')
            members = set()
            for target in targets:
                if Path(target).suffix in ('.sln', '.slnx'):
                    output = self.cmd(['dotnet', 'sln', target, 'list'], u, capture=True)
                    for line in output.splitlines():
                        if line.strip().endswith(('.csproj', '.fsproj', '.vbproj')):
                            members.add(str(Path(target).parent / line.strip().replace('\\', '/')))
                else: members.add(target)
                self.cmd(['dotnet', 'build', target, '-c', 'Release', '--nologo', '-m:2', '/nodeReuse:false'], u)
            for p in sorted(members):
                data = self.cmd(['dotnet', 'msbuild', p, '-nologo', '-p:Configuration=Release',
                    '-getProperty:IsTestProject,IsTestingPlatformApplication,OutputType,TargetFramework,TargetFrameworks',
                    '-getItem:PackageReference'], u, capture=True)
                meta = json.loads(data[data.index('{'):])
                props = meta['Properties']
                packages = {x['Identity'].lower() for x in meta.get('Items', {}).get('PackageReference', [])}
                test = props.get('IsTestProject', '').lower() == 'true' or bool(packages & {'microsoft.net.test.sdk', 'tunit', 'mstest', 'microsoft.testing.platform'})
                mtp = props.get('IsTestingPlatformApplication', '').lower() == 'true' or 'tunit' in packages
                if test:
                    dest = f'tests/{uid}/{hashlib.sha256(p.encode()).hexdigest()[:10]}'
                    if mtp:
                        argv = ['dotnet', 'run', '--project', p, '-c', 'Release', '--no-build', '--', '--no-ansi']
                        # MTP's default nonzero exit code for zero tests is preserved.
                        u['test'].append(command(argv, u['root']))
                    else:
                        argv = ['dotnet', 'test', p, '-c', 'Release', '--no-build', '--nologo',
                                '--logger', 'trx', '--results-directory', str(self.work / dest)]
                        u['test'].append(command(argv, u['root'], require_tests=dest))
                if 'benchmarkdotnet' in packages and props.get('OutputType', '').lower() == 'exe':
                    benchmark_dir = str(Path('benchmarks') / uid / Path(p).stem)
                    u['benchmark'].append(command(['dotnet', 'run', '--project', p, '-c', 'Release', '--no-build', '--',
                        '--filter', '*', '--job', 'short', '--exporters', 'json', '--artifacts', str(self.work / 'benchmarks' / uid / Path(p).stem)],
                        u['root'], adapter='c_sharp_dot_net', require_benchmarks=benchmark_dir))
        elif kind == 'python':
            python = self.work / 'venvs' / uid / 'bin/python'
            self.cmd([u.get('python', 'python3'), '-m', 'venv', str(python.parent.parent)], u)
            manifest = directory / 'pyproject.toml'
            meta = tomllib.loads(text(manifest)) if manifest.exists() else {}
            dependency = directory / 'requirements.txt'
            if dependency.exists(): self.cmd([str(python), '-m', 'pip', 'install', '-r', 'requirements.txt'], u)
            if (directory / 'setup.py').exists() or 'build-system' in meta:
                extras = [name for name in ('test', 'tests', 'dev') if name in meta.get('project', {}).get('optional-dependencies', {})]
                requirement = '.' + ('[' + ','.join(extras) + ']' if extras else '')
                self.cmd([str(python), '-m', 'pip', 'install', '-e', requirement], u)
            elif meta.get('project', {}).get('dependencies'):
                self.cmd([str(python), '-m', 'pip', 'install', *meta['project']['dependencies']], u)
            for name in ('requirements-test.txt', 'requirements-dev.txt'):
                if (directory / name).exists(): self.cmd([str(python), '-m', 'pip', 'install', '-r', name], u)
            groups = meta.get('dependency-groups', {})
            for name in ('dev', 'test'):
                deps = [d for d in groups.get(name, []) if isinstance(d, str)]
                if deps: self.cmd([str(python), '-m', 'pip', 'install', *deps], u)
            sources = u.get('files', [])
            # Compile tracked source only, without importing applications or running examples.
            for start in range(0, len(sources), 100):
                self.cmd([str(python), '-m', 'py_compile', *[str(self.root / p) for p in sources[start:start+100]]], u)
            tests = [p for p in sources if Path(p).name.startswith('test_') or Path(p).name.endswith('_test.py')]
            benchmark_tests = [p for p in tests if re.search(r'\bbenchmark\s*\(', text(self.root / p)) or 'pytest_benchmark' in text(self.root / p)]
            if tests or 'pytest' in meta.get('tool', {}):
                self.cmd([str(python), '-m', 'pip', 'install', 'pytest', *(['pytest-benchmark'] if benchmark_tests else [])], u)
                argv = [str(python), '-m', 'pytest', '-q', '--junitxml=' + str(self.work / f'{uid}-tests.xml')]
                if benchmark_tests: argv.append('--benchmark-skip')
                u['test'].append(command(argv, u['root']))
            if benchmark_tests:
                u['benchmark'].append(command([str(python), '-m', 'pytest', '-q', '--benchmark-only',
                    '--benchmark-json=' + str(self.work / f'{uid}-benchmarks.json')], u['root'], adapter='python_pytest'))
            candidates = [p for p in sources if Path(p).name.startswith(('bench_', 'benchmark_')) and p not in benchmark_tests]
            if candidates: u['unresolved_benchmarks'] = candidates
        elif kind == 'zig':
            help_text = self.cmd(['zig', 'build', '--help'], u, capture=True)
            steps = set(re.findall(r'^  ([\w-]+)\s', help_text.split('General Options:')[0], re.M))
            self.cmd(['zig', 'build', '-Doptimize=ReleaseFast', '-j2'], u)
            if 'test' in steps: u['test'].append(command(['zig', 'build', 'test', '-j2', '--summary', 'all'], u['root']))
            for name in ('bench', 'benchmark', 'benchmarks'):
                if name in steps:
                    u['benchmark'].append(command(['zig', 'build', name, '-Doptimize=ReleaseFast', '-j1', '--summary', 'all'], u['root']))
                    break
        elif kind == 'cmake':
            build = str(self.work / 'build' / uid)
            query = Path(build) / '.cmake/api/v1/query'
            query.mkdir(parents=True, exist_ok=True)
            (query / 'codemodel-v2').touch()
            self.cmd(['cmake', '-S', '.', '-B', build, '-DCMAKE_BUILD_TYPE=Release', '-DBUILD_TESTING=ON'], u)
            self.cmd(['cmake', '--build', build, '--parallel', '2'], u)
            data = json.loads(self.cmd(['ctest', '--test-dir', build, '--show-only=json-v1'], u, capture=True))
            registered = data.get('tests', [])
            bench_names = []
            for t in registered:
                labels = [p['value'] for p in t.get('properties', []) if p['name'] == 'LABELS']
                if any(re.search(r'bench(mark)?', str(label), re.I) for label in labels): bench_names.append(t['name'])
            test_names = [t['name'] for t in registered if t['name'] not in bench_names]
            for stage, names in [('test', test_names), ('benchmark', bench_names)]:
                if names:
                    expression = '^(' + '|'.join(re.escape(name) for name in names) + ')$'
                    u[stage].append(command(['ctest', '--test-dir', build, '--output-on-failure', '--no-tests=error',
                        '-R', expression, '--output-junit', str(self.work / f'{uid}-{stage}.xml')], u['root']))
            # Google Benchmark executables are identified from the evaluated link graph.
            # Merely naming a target "bench" does not tell us how to run it.
            reply = Path(build) / '.cmake/api/v1/reply'
            for model in reply.glob('codemodel-v2-*.json'):
                for configuration in read_json(model)['configurations']:
                    for target in configuration['targets']:
                        info = read_json(reply / target['jsonFile'])
                        links = ' '.join(fragment['fragment'] for fragment in info.get('link', {}).get('commandFragments', []))
                        if info.get('type') == 'EXECUTABLE' and re.search(r'(?:libbenchmark|(?:^|\s)-lbenchmark)', links):
                            artifact = info.get('artifacts', [{}])[0].get('path')
                            if artifact:
                                u['benchmark'].append(command([str(Path(build) / artifact),
                                    '--benchmark_out=' + str(self.work / f'{uid}-{info["name"]}-benchmarks.json'),
                                    '--benchmark_out_format=json'], u['root'], adapter='cpp_google'))
        elif kind == 'meson':
            build = str(self.work / 'build' / uid)
            self.cmd(['meson', 'setup', build, '--buildtype=release'], u)
            self.cmd(['meson', 'compile', '-C', build, '-j', '2'], u)
            for stage, flag in [('test', '--tests'), ('benchmark', '--benchmarks')]:
                data = json.loads(self.cmd(['meson', 'introspect', build, flag], u, capture=True))
                if data: u[stage].append(command(['meson', 'test', '-C', build, '--print-errorlogs', *(['--benchmark'] if stage == 'benchmark' else [])], u['root']))
        elif kind == 'make':
            self.cmd(['make', '-j2'], u)
            # make's database evaluates included/conditional targets. -q must not execute them.
            output = self.cmd(['python3', '-c',
                'import subprocess,sys; p=subprocess.run(["make","-qp"],stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True); print(p.stdout); sys.exit(0 if p.returncode in (0,1) else p.returncode)'], u, capture=True)
            targets = set(re.findall(r'^([A-Za-z][\w-]*):(?:[^=]|$)', output, re.M))
            for stage, choices in [('test', ('test', 'check')), ('benchmark', ('bench', 'benchmark', 'benchmarks'))]:
                for name in choices:
                    if name in targets:
                        u[stage].append(command(['make', name], u['root'])); break
        elif kind == 'rust':
            self.cmd(['cargo', 'build', '--workspace', '--all-targets', '--locked'], u)
            u['test'].append(command(['cargo', 'test', '--workspace', '--locked'], u['root']))
            meta = json.loads(self.cmd(['cargo', 'metadata', '--no-deps', '--format-version=1'], u, capture=True))
            if any('bench' in target['kind'] for package in meta['packages'] for target in package['targets']):
                u['benchmark'].append(command(['cargo', 'bench', '--workspace', '--locked'], u['root'], adapter='rust_criterion'))
        elif kind == 'go':
            self.cmd(['go', 'build', './...'], u)
            files = [p for p in tracked(self.root) if under(p, Path(u['root'])) and p.name.endswith('_test.go')]
            if files: u['test'].append(command(['go', 'test', './...'], u['root']))
            if any(re.search(r'^func Benchmark\w+\(', text(self.root / p), re.M) for p in files):
                u['benchmark'].append(command(['go', 'test', '-run=^$', '-bench=.', '-benchmem', './...'], u['root'], adapter='go_bench'))
        elif kind == 'node':
            scripts = read_json(directory / 'package.json', {}).get('scripts', {})
            if (directory / 'pnpm-lock.yaml').exists(): install, run = ['pnpm', 'install', '--frozen-lockfile'], ['pnpm', 'run']
            elif (directory / 'yarn.lock').exists(): install, run = ['yarn', 'install', '--frozen-lockfile'], ['yarn', 'run']
            elif (directory / 'package-lock.json').exists(): install, run = ['npm', 'ci'], ['npm', 'run']
            else: install, run = ['npm', 'install', '--package-lock=false'], ['npm', 'run']
            self.cmd(install, u)
            if 'build' in scripts: self.cmd([*run, 'build'], u)
            if 'test' in scripts: u['test'].append(command([*run, 'test'], u['root']))
            for name in ('bench', 'benchmark'):
                if name in scripts: u['benchmark'].append(command([*run, name], u['root'])); break
        else: raise ValueError(f'Unsupported adapter: {kind}')

    def stage(self, name):
        if name == 'discover':
            self.report['stages'] = {}
            plan = discover(self.root)
            self.plan_path.write_text(json.dumps(plan, indent=2))
            print(json.dumps(plan, indent=2))
            self.report['stages']['discover'] = 'passed'
            self.save()
            return
        plan = read_json(self.plan_path)
        predecessor = {'build': 'discover', 'test': 'build', 'benchmark': 'test'}[name]
        if self.report['stages'].get(predecessor) not in ('passed', 'skipped'):
            raise RuntimeError(f'{name} requires successful {predecessor}')
        self.report['stages'][name] = 'running'; self.save()
        try:
            count = 0
            for u in plan['projects']:
                if name == 'build':
                    self.configure(u)
                    count += 1
                for c in u[name]:
                    self.run(c, name, u['id']); count += 1
            self.plan_path.write_text(json.dumps(plan, indent=2))
            self.report['stages'][name] = 'passed' if count else 'skipped'
            if not count: print(f'{name}: skipped; no runnable {name} targets detected')
            if name == 'benchmark':
                unresolved = [p for u in plan['projects'] for p in u.get('unresolved_benchmarks', [])]
                if unresolved:
                    self.report['unresolved_benchmarks'] = unresolved
                    print('::warning::Benchmark scripts need explicit argv in .forgejo/polyglot-ci.json: ' + ', '.join(unresolved))
            self.save()
        except Exception as error:
            self.report['stages'][name] = 'failed'
            self.report['error'] = str(error); self.save()
            raise

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=('discover', *STAGES, 'all'))
    parser.add_argument('--root', type=Path, default=Path.cwd())
    args = parser.parse_args()
    runner = Runner(args.root)
    try:
        for stage in ('discover', *STAGES) if args.stage == 'all' else (args.stage,): runner.stage(stage)
    except Exception as error:
        print(f'::error::{error}', file=sys.stderr)
        return 1
    return 0

if __name__ == '__main__': sys.exit(main())
