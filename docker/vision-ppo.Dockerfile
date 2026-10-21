# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

# OVRTX training runtime, isolated from Kit's USD shared libraries.
ARG SIM_IMAGE=so101-typing-public:dev
FROM ${SIM_IMAGE}
USER root
COPY runtime/renderer.lock /tmp/renderer.lock
RUN env -u PYTHONPATH /workspace/isaaclab/_isaac_sim/kit/python/bin/python3 -m venv --system-site-packages /opt/vision \
    && /opt/vision/bin/python -m pip download --no-cache-dir --no-deps --only-binary=:all: \
       --require-hashes --index-url https://pypi.nvidia.com --dest /tmp/renderer-wheels -r /tmp/renderer.lock \
    && /opt/vision/bin/python -m pip install --no-cache-dir --index-url https://pypi.org/simple \
       usd-core==25.5.1 /tmp/renderer-wheels/*.whl \
    && rm -rf /tmp/renderer-wheels /tmp/renderer.lock
ENV PYTHONPATH="" OVRTX_SKIP_USD_CHECK=1 PYTHONUNBUFFERED=1
COPY docker/vision-entrypoint.sh /opt/vision/entrypoint.sh
COPY source/so101_typing /workspace/isaaclab/source/so101_typing
COPY scripts /workspace/isaaclab/scripts
COPY configs /workspace/isaaclab/configs
ENTRYPOINT ["/bin/bash", "/opt/vision/entrypoint.sh"]
