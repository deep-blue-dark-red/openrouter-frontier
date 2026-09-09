# Forgejo polyglot CI

One ordered pipeline for a repository, including repositories with several build systems:

`discover → prepare toolchains → compile all projects → test all detected suites → benchmark`

The orchestrator and its regression tests are Python 3.11+ and use the standard
library. Repository tests remain in their native languages and frameworks.

| Project | Build | Test discovery and execution | Benchmark discovery |
| --- | --- | --- | --- |
| .NET / C#, F#, VB | solution (`.slnx` preferred) or projects, Release | evaluated MSBuild properties and package references; VSTest or MTP/TUnit | executable projects referencing BenchmarkDotNet |
| Python | isolated venv, declared dependencies, compile tracked Python source | pytest, which also executes unittest suites; no fallback after failure | pytest-benchmark fixtures; bespoke scripts require explicit commands |
| Zig | `zig build`, ReleaseFast | advertised `test` build step | advertised `bench`, `benchmark`, or `benchmarks` step |
| C / C++ with CMake | configure and build Release | CTest JSON registry; zero-test execution errors | CTest benchmark labels and Google Benchmark executable link metadata |
| C / C++ with Meson | setup and compile Release | Meson's test registry | Meson's benchmark registry |
| Make | default target | evaluated `test` or `check` target | evaluated benchmark target |
| Rust | Cargo workspace build | `cargo test`, never nextest | Cargo benchmark targets |
| Go | `go build` | Go test files | Go benchmark functions |
| JavaScript / TypeScript | declared package manager and build script | declared test script | declared benchmark script |

Detection considers tracked files, not the Forgejo language badge or files left
over from earlier jobs. A root build system owns its nested subdirectories;
different build systems are independent. Vendored code, generated code, fixtures,
and build output are excluded by default (see `SKIP` in pipeline.py).
An unsupported project or a documentation-only repository reports skipped stages,
not a passed test suite. A detected failing command never falls back to another
runner. VSTest must produce a TRX with executed tests. pytest's nonzero no-tests
exit and MTP's no-tests failure are preserved.

## Repository contract

Most repositories need only the managed workflow and vendored runner files.
Unusual layouts, hardware requirements, private dependency installation, custom
benchmark arguments, or custom test executables can use `.forgejo/polyglot-ci.json`:

```json
{
  "version": 1,
  "projects": [
    {
      "kind": "custom",
      "root": ".",
      "build": [{"argv": ["make", "release"]}],
      "test": [{"argv": ["./out/check"], "timeout": 600}],
      "benchmark": [{"argv": ["./out/measure", "--json"], "timeout": 1800}]
    }
  ]
}
```

`projects` replaces automatic project selection. Commands are argument arrays,
without implicit shell expansion. An intentional shell script can be invoked
with `["bash", "scripts/ci-test.sh"]`. Its exit code must reflect real failures.
`cwd` defaults to the project root and cannot escape the repository.
For automatic adapters, list a kind/root pair without commands. The dotnet
adapter additionally needs `targets`; the Python adapter needs `files` if using
explicit project selection. `exclude` adds glob exclusions to automatic discovery.
`toolchains.dotnet` accepts exact SDK versions; `toolchains.zig` accepts an exact
Zig version. Normal .NET global.json roll-forward policies are honored by the
installed SDK selection; an unavailable pinned SDK fails preparation.

Arbitrary benchmark scripts are reported as unresolved with their paths. They
are not executed by filename alone: arguments, datasets, hardware, and output
contracts must be supplied by the repository. BenchmarkDotNet uses a short job;
this is an initial CI sample, not evidence of a statistically significant
performance improvement. Legacy workflows can still create host contention.

## Storage and execution

All new runner workspaces, toolchain installations, dependency caches, temporary
files, and preserved results use `/mnt/commons/git-ci/polyglot/`. The workflow fails before
checkout if its runner workspace is outside that volume. Reports are retained at
`results/<owner>/<repository>/<run-id>/<attempt>/`, including command logs, stage
status, test reports, benchmark files, commit identity, and host information.
Build binaries remain in the workspace on that same volume.

The dedicated `commons-ci` host runner executes one job at a time. A volume-wide
lock also serializes preparation, builds, tests, and benchmarks across managed
jobs. This avoids overlap within this pipeline; unrelated legacy jobs are not
controlled by this lock. Workflows run on pushes and manual dispatch; a privileged
host runner does not automatically run untrusted pull-request code.

The Forgejo server's own uploaded-artifact and Actions-log storage is configured
separately by an administrator. Until it is confirmed at the required location,
the workflow preserves reports directly on the external volume instead of creating
another artifact copy through the server.

## Maintaining the shared implementation

This repository is the canonical source. Each consumer gets the same runner
files plus a SHA-256 manifest. Vendoring makes the exact CI implementation
reviewable in each commit, requires no cross-repository read token in jobs, and
works for private repositories. Fleet rollout changes only the managed files,
uses Forgejo blob SHAs for updates, and records one commit per repository.
Existing workflows are preserved so another agent's concurrent edits are not lost.

Run the orchestrator tests with:

```sh
python3 -m unittest discover -s tests -v
```

The suite includes a real C/C++ build with separately registered CTest correctness
and benchmark targets, plus failure-gating, timeout, traversal, mixed-language,
fixture-exclusion, MTP selection, and zero-test regression cases.

Upstream references: [Forgejo workflow syntax](https://forgejo.org/docs/v13.0/user/actions/reference/),
[MSBuild evaluated metadata](https://learn.microsoft.com/en-us/visualstudio/msbuild/evaluate-items-and-properties),
[CTest](https://cmake.org/cmake/help/latest/manual/ctest.1.html),
[Zig build steps](https://ziglang.org/learn/build-system/), and
[Forgejo storage](https://forgejo.org/docs/v13.0/admin/setup/storage/).
