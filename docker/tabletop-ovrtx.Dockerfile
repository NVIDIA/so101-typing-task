# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

# Optional RTX renderer: keep its USD libraries separate from Isaac Sim's Python.
ARG SIM_IMAGE=so101-typing-public:dev
FROM ${SIM_IMAGE}
USER root
RUN env -u PYTHONPATH /workspace/isaaclab/_isaac_sim/kit/python/bin/python3 -m venv /opt/ovrtx \
    && /opt/ovrtx/bin/python -m pip install --no-cache-dir \
       --extra-index-url https://pypi.nvidia.com ovrtx==0.5.1.385782 ovstage==0.2.1.385922 numpy==2.3.1 Pillow==11.3.0
ENV PYTHONPATH="" OVRTX_SKIP_USD_CHECK=1 PYTHONUNBUFFERED=1
ENTRYPOINT ["/opt/ovrtx/bin/python"]
