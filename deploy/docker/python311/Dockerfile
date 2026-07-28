FROM python:3.11-slim-bookworm

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

# DockerProcessRunner always invokes non-interactive Bash explicitly.
ENTRYPOINT []
CMD ["/bin/bash", "--noprofile", "--norc"]
