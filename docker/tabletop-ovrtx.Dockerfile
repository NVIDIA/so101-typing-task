# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

# Optional RTX renderer: keep its USD libraries separate from Isaac Sim's Python.
ARG SIM_IMAGE=so101-typing-public:dev
FROM ${SIM_IMAGE}
USER root
COPY runtime/renderer.lock /tmp/renderer.lock
RUN env -u PYTHONPATH /workspace/isaaclab/_isaac_sim/kit/python/bin/python3 -m venv /opt/ovrtx \
    && /opt/ovrtx/bin/python -m pip download --no-cache-dir --no-deps --only-binary=:all: \
       --require-hashes --index-url https://pypi.nvidia.com --dest /tmp/renderer-wheels -r /tmp/renderer.lock \
    && /opt/ovrtx/bin/python -m pip install --no-cache-dir --index-url https://pypi.org/simple \
       /tmp/renderer-wheels/*.whl numpy==2.3.1 Pillow==11.3.0 \
    && rm -rf /tmp/renderer-wheels /tmp/renderer.lock
ENV PYTHONPATH="" OVRTX_SKIP_USD_CHECK=1 PYTHONUNBUFFERED=1
ENTRYPOINT ["/opt/ovrtx/bin/python"]
