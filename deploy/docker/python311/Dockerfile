FROM mcr.microsoft.com/powershell:7.6-debian-12

USER root

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        ca-certificates \
        git \
        python3 \
        python3-pip \
        python3-venv \
        ripgrep \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.lock /tmp/requirements.lock
RUN python3 -m venv /opt/venv \
    && /opt/venv/bin/python -m pip install --no-cache-dir --upgrade pip \
    && /opt/venv/bin/python -m pip install --no-cache-dir -r /tmp/requirements.lock \
    && rm /tmp/requirements.lock \
    && chmod -R a+rX /opt/venv

ENV PATH="/opt/venv/bin:${PATH}"

# DockerProcessRunner 总是显式传入 pwsh；清除基础镜像入口以避免重复解释命令。
ENTRYPOINT []
CMD ["pwsh", "-NoLogo", "-NoProfile", "-NonInteractive"]
