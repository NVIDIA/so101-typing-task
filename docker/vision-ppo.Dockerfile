# OVRTX training runtime, isolated from Kit's USD shared libraries.
ARG SIM_IMAGE=so101-typing-public:dev
FROM ${SIM_IMAGE}
USER root
RUN env -u PYTHONPATH /workspace/isaaclab/_isaac_sim/kit/python/bin/python3 -m venv --system-site-packages /opt/vision \
    && /opt/vision/bin/python -m pip install --no-cache-dir --extra-index-url https://pypi.nvidia.com \
       usd-core==25.5.1 ovrtx==0.5.1.385782 ovstage==0.2.1.385922
ENV PYTHONPATH="" OVRTX_SKIP_USD_CHECK=1 PYTHONUNBUFFERED=1
COPY docker/vision-entrypoint.sh /opt/vision/entrypoint.sh
COPY source/so101_typing /workspace/isaaclab/source/so101_typing
COPY scripts /workspace/isaaclab/scripts
COPY configs /workspace/isaaclab/configs
ENTRYPOINT ["/bin/bash", "/opt/vision/entrypoint.sh"]
