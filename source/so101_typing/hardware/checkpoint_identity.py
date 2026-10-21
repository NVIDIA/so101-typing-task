# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""Pin published policy/environment pairs without inferring unknown provenance.

These checkpoints do not embed their training environment identity. Known
releases must retain their exact published pair. Unknown checkpoints still use
the supplied environment's contract checks; this does not prove training linkage.
Training/resume intentionally does not use this evaluation/deployment guard.
"""
import hashlib
from pathlib import Path


REFERENCE_PAIR = (
    '239d93c99cd282e9add019f4ac952407c97fde2879821103a2aa7bd7a307a15f',
    'd4766dacb28a4ffc0fc75efe8e5abed162e656b338b110e8d498cc808e31ef8b',
)
# Same immutable payload hashes recorded in scripts/download_benchmark_artifacts.sh.
# That script refuses to fetch them until the dataset card declares a license.
RELEASED_ENVIRONMENTS = {
    REFERENCE_PAIR[0]: REFERENCE_PAIR[1],  # July AnchorBench 19k
    'cc7f128e15a282a6f2fcc68e97f67c2e6f81bf6e03347b6a3cbf0b9a02d072a4':
        'd14d60d19554326d66cbffc175c42166232968380980574ae36c196c9a0f7acb',  # USD 19k
    'f7fdd19b4c4b971c5363391292b4e3e7bbee602a195d7630814fa2a23d71e7a4':
        'c61cba29921c2187f43591561ab80beb2980c7436e77709019d3bec04da34405',  # public P1A
    'ea6c940f8a57797acaaca845bfb4b7cf2f1780f3d9571536b20998cbf114e0a8':
        '4942a720f3c2cc43aae6d66973c260aeec56a6e9891d5bb8b57febdf65e772bd',  # public Transit15
    '6af850866096ec6b6ad6b202c8ac9417681efa0ebe71f4e8e8fdcf57d393f225':
        '72bb3f80cd41da4930c0fe5de54707ba3dfa1ee772f9745afc128fcd01f8db32',  # September
}


def validate_pair(checkpoint_hash, environment_hash):
    expected = RELEASED_ENVIRONMENTS.get(checkpoint_hash)
    if expected is not None and environment_hash != expected:
        raise ValueError('published checkpoint requires its matching environment; '
                         'the supplied environment SHA-256 differs from the release')


def validate_files(checkpoint, environment):
    validate_pair(hashlib.sha256(Path(checkpoint).read_bytes()).hexdigest(),
                  hashlib.sha256(Path(environment).read_bytes()).hexdigest())
