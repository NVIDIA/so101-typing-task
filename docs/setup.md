# Runtime setup

## Requirements

- Linux x86-64, Docker, NVIDIA GPU driver and NVIDIA Container Toolkit.
- An RTX-capable NVIDIA GPU for OVRTX; GPU memory requirements depend on the
  environment count. Start with the two-environment smoke preset.
- An X11 desktop (`DISPLAY` and `XAUTHORITY`) for the interactive RTX viewer.
  Headless evaluation and MP4 export do not need a desktop.
- Internet access for the initial image build or configured release downloads.
  The currently built tutorial image is approximately 50 GiB uncompressed
  (shared layers may reduce additional storage). Allow additional room for
  builds, checkpoints and videos; inspect disk space with `./so101 debug` and
  `docker system df` before building.
- Hardware deployment additionally requires a SO-101, wrist camera, keyboard,
  USB access, motor calibration and a measured simulation mapping/rest pose.

The tutorial is being validated on Linux with an RTX PRO 6000 Blackwell Max-Q
(approximately 96 GiB VRAM), driver 595.91.07. This is a tested configuration,
not a claimed minimum. Results and memory use on other GPUs are not yet
established. First startup compiles kernels and initializes RTX; expect minutes
before the first update. Do not interpret a smoke test as convergence evidence.


## Install

```bash
git clone <YOUR_REPOSITORY_URL>
cd so101-keyboard-typing-internal
git checkout gtc_berlin
./so101 setup
./so101 debug --viewer
```

`setup` uses the repository Dockerfiles. A distributor can configure reviewed,
digest-pinned prebuilt images in [releases.json](../configs/tutorial/releases.json)
so users pull the runtime instead. This checkout does not publish images or
models automatically. Run `./so101 setup --build` after editing simulator
source to rebuild the tutorial runtime. Hardware setup uses `uv` and the pinned
[runtime/hardware.lock](../runtime/hardware.lock):

```bash
./so101 setup --hardware
```

See [NVIDIA Container Toolkit installation](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)
and [uv installation](https://docs.astral.sh/uv/getting-started/installation/).
The host driver is required even though simulation dependencies live in Docker.
