# Security boundaries

Load tensor checkpoints only from sources you trust. Project checkpoint readers
use PyTorch's restricted `weights_only=True` deserializer. RSL-RL readers enforce
the same restriction through `TORCH_FORCE_WEIGHTS_ONLY_LOAD=1`, which stays enabled
for the process and its children. A rejected checkpoint must not be retried with
unsafe pickle loading or an automatically extended class allowlist.

Restricted deserialization does not establish model provenance, prevent memory
exhaustion, or prove that the resulting actions are safe. Verify the publisher and
the expected checkpoint hash independently of files supplied with a model bundle.

Exported TorchScript actors are executable programs. Create them locally from a
trusted checkpoint using this repository's exporter, or obtain them from a trusted
publisher. The actor and source hashes in bundle metadata detect changes relative
to that metadata; they do not authenticate a publisher when an attacker can replace
both the program and the metadata.

Run simulation and rendering on an isolated development machine without sensitive
files in mounted directories. Launchers drop Linux capabilities and set
`no-new-privileges`; GPU workloads use a private IPC namespace with bounded shared
memory. These controls do not isolate the GPU driver or block outbound networking.
NVIDIA Python wheels are downloaded exclusively from NVIDIA's index with pinned
hashes before their public dependencies are resolved separately from PyPI. This is
not a complete hash lock for every transitive Python or operating-system package.

Interactive X11 viewing grants access to the host
desktop session; a read-only socket mount does not restrict X11 protocol access.
Use headless recording when desktop access is unnecessary. Only expose a dedicated,
non-sensitive keyboard application to the robot. Keep the workspace and serial
connection under operator control and retain the deployment preflight, action
bounds, stop controls and torque-off checks.
