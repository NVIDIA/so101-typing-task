# Runtime provenance

Public Isaac Lab source ancestor:
https://github.com/ooctipus/IsaacLab/commit/6f991d4becf764b151e0a1c775561ddd9c406f72

Benchmark source snapshot: `1b8cb7e8176021685325724b62c41b1fe394c25a`.
The public source ancestor is fetched by the Dockerfile. The task source is
already included in this repository. `benchmark-runtime.patch` carries the
additional Isaac Lab / Newton integration and shared DexSuite changes required
by that snapshot. It preserves upstream copyright and license headers.

The patch was generated with `git diff PUBLIC_ANCESTOR BENCHMARK_SNAPSHOT` over:
- `source/isaaclab`
- `source/isaaclab_newton`
- `source/isaaclab_rl`
- `source/isaaclab_tasks/isaaclab_tasks/core/dexsuite`, excluding `config/`

Changes include articulation actuator support, Newton manager configuration,
debug state recording, and shared task behavior. The public build fetches no
private repository.

Pinned image digest and Python dependency versions are in
`docker/typing-public.Dockerfile`. Build success does not establish solver or
checkpoint equivalence; GPU probes and evaluation remain required.

The public calibration JSON omits machine-specific provenance paths. Its own
checksum is `PUBLIC_CALIBRATION_DATASET_SHA256`; the historical dataset identity
remains in the frozen checkpoint contract. Numerical measurements and the target
map are unchanged.

The keyboard NPZ's embedded source filename is also sanitized. Its 20 geometry
and texture-coordinate array payloads are unchanged. Source-identity checks
recognize only the exact original and sanitized asset hashes (and the resulting
drawing metadata hashes), preserving existing recording and alignment evidence
without accepting changes to geometry or other source files.
