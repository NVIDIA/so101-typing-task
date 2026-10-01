# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

# Public base pinned by digest; no workstation image or private source required.
FROM nvcr.io/nvidia/isaac-lab@sha256:970621075d00059c309f847fa835709de3fa634537407a7009c586c97893d218
USER root
WORKDIR /workspace/isaaclab
# Replace the older image's Python sources with the benchmark's public ancestor.
ADD https://codeload.github.com/ooctipus/IsaacLab/tar.gz/6f991d4becf764b151e0a1c775561ddd9c406f72 /tmp/isaaclab.tar.gz
RUN echo '9d2a9a9ea37bff991a32c8ce597a13b7915211a6c4fcd88732a62efa6bf31d84  /tmp/isaaclab.tar.gz' | sha256sum -c -
RUN rm -rf /workspace/isaaclab/source /workspace/isaaclab/scripts && tar -xzf /tmp/isaaclab.tar.gz --strip-components=1 -C /workspace/isaaclab
COPY runtime/benchmark-runtime.patch /tmp/benchmark-runtime.patch
RUN git apply /tmp/benchmark-runtime.patch
RUN ./isaaclab.sh -p -m pip install --no-cache-dir \
    hydra-core==1.3.4 llvmlite==0.46.0 numba==0.63.1 numpy==2.3.1 rsl-rl-lib==5.4.1 \
    && ./isaaclab.sh -p -m pip install --no-cache-dir --no-deps \
    'newton[sim] @ git+https://github.com/ooctipus/newton.git@5205aa49fb900e124f06122bf7e82ed7686a1fde' \
    && ./isaaclab.sh -p -m pip install --no-cache-dir --extra-index-url https://pypi.nvidia.com \
    warp-lang==1.15.0.dev20260626 newton-usd-schemas==0.3.1
ADD https://huggingface.co/datasets/nvidia/Anchor-Lab/resolve/647edd5787cd764cdc041103ad282dc59214d919/robot_assets/so101_no_camera_new_calib.usd /opt/so101/assets/so101.usd
RUN echo 'c6c82840925ace388b0ff0acb7d8538c2b419d92fbe01dc70fe833b974d6d462  /opt/so101/assets/so101.usd' | sha256sum -c -
RUN chmod 644 /opt/so101/assets/so101.usd
ENV SO101_ROBOT_USD=/opt/so101/assets/so101.usd
# Preserve the published Isaac Lab namespace while keeping project code together.
# Drop deprecated per-framework launchers supplied by the ancestor; the unified
# launchers dispatch to train_rsl_rl.py and play_rsl_rl.py instead.
RUN rm -rf /workspace/isaaclab/source/isaaclab_tasks/isaaclab_tasks/core/dexsuite/config/so101 \
    /workspace/isaaclab/scripts/so101_homing \
    /workspace/isaaclab/scripts/reinforcement_learning/rsl_rl/train.py \
    /workspace/isaaclab/scripts/reinforcement_learning/rsl_rl/play.py
COPY source/ /workspace/isaaclab/source/
COPY scripts/ /workspace/isaaclab/scripts/
COPY pyproject.toml /workspace/isaaclab/pyproject.toml
COPY LICENSE THIRD_PARTY_NOTICES.md /workspace/isaaclab/
COPY licenses/ /workspace/isaaclab/licenses/
RUN ./isaaclab.sh -p -m pip install --no-deps --no-build-isolation --editable .
ENV ACCEPT_EULA=Y NO_NUCLEUS=Y OMNI_KIT_ALLOW_ROOT=1 PYTHONUNBUFFERED=1
ARG SOURCE_SHA256
LABEL org.so101.source-sha256=${SOURCE_SHA256}
ENTRYPOINT ["/workspace/isaaclab/isaaclab.sh"]
