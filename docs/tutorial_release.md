# Distributing the tutorial and selected checkpoints

The repository contains source, configurations, documentation and manifests.
JSON metadata for the NEWTON vision and ISAAC LAB 3.0 state checkpoints stays
under `models/`. The weight files are not tracked. Product Legal request 2132
is still in review; see [models/README.md](../models/README.md).
Generated weights, recordings, reports and local profiles stay ignored.
The setup and bundle tools never upload files. Existing downloaded artifacts are
not copied into image build contexts (`.dockerignore`).

## Runtime images

Use a versioned registry namespace to build the two runtime images:

```bash
scripts/tutorial/build_release_images.sh registry.example.org/team/so101 v1
```

This builds images locally and prints their IDs. It does not push them. After a
maintainer explicitly publishes the images, record their immutable references
in `configs/tutorial/releases.json`:

```json
{
  "schema_version": 1,
  "images": {
    "base": "REGISTRY/so101-base@sha256:ACTUAL_DIGEST",
    "vision": "REGISTRY/so101-vision@sha256:ACTUAL_DIGEST"
  },
  "models": {}
}
```

The values above are placeholders, not working releases. Review upstream asset
and image redistribution terms before selecting a distribution channel. Runtime
images include simulation assets and code, but no learned policy weights. Users
can run `./so101 setup --release path/to/releases.json`; digest-pinned images are
pulled and tagged for the launchers. Building from source remains available when
no release images are configured. `debug` exercises the installed runtime.

## Model bundles

To package a checkpoint locally:

```bash
./so101 bundle pack --checkpoint output/my-vision/model_4999.pt \
  --archive output/releases/vision-demo.zip
```

The ZIP contains `checkpoint.pt`, `environment.json`, `agent.json`, `run.json`,
the evaluation camera preset, and available task/asset metadata. `bundle.json`
records per-file hashes. A separate `.sha256.json` records the archive hash.
The archive stays under output and is not uploaded or committed. Keep model
licensing and evaluation evidence alongside any independently distributed copy.

A distributor who already hosts a trusted bundle can add a `models` entry:

```json
{"vision-demo": {"url": "https://HOST/vision-demo.zip", "sha256": "ACTUAL_ARCHIVE_SHA256"}}
```

Users download with `./so101 download --model vision-demo --manifest releases.json`.
Downloads require HTTPS and a matching archive hash. Extraction validates member
hashes, rejects unsafe paths and limits archive size. Checksums provide integrity,
not trust in a model's source. Runtime checkpoint loading must use trusted weights.
No model URL is fabricated or enabled in the default empty manifest.

For offline sharing, `bundle import --archive FILE --name NAME --sha256 HASH`
uses the same validation and installs under `.artifacts/models/NAME`. Replay
accepts either a model name, checkpoint path, or run directory. Saved settings
remain paired with the model. Hardware calibration is deliberately local and
is not included in a portable model bundle.

## Verify a release on a fresh machine

1. Clone only tracked source and install/pull the runtime.
2. Run debug and the two-environment teacher-to-vision smoke workflow.
3. Import a test bundle and replay it both interactively and headlessly.
4. Evaluate a fixed phrase and a random-sequence bank; record protocols separately.
5. Check hardware profiles without motion. Perform a physical trial only with
   that machine's calibrated mapping and rest pose.
6. Record tested OS, GPU, driver, peak memory, image sizes and first/warm startup
   timings. Do not infer minimum requirements from a large workstation GPU.
7. Check `git diff --cached --name-only`: no weights, archives, local profiles,
   output logs, videos, PDFs, or PowerPoint files belong in the source commit.

The existing source checks remain in place; this change adds no publishing job
or automatic model upload.
