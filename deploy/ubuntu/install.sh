#!/usr/bin/env bash
set -euo pipefail

CONFIG=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --config) CONFIG="${2:?--config requires a path}"; shift 2 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done
if [[ -z "${CONFIG}" || ! -f "${CONFIG}" ]]; then
  echo "usage: sudo ./deploy/ubuntu/install.sh --config config.yml" >&2
  exit 2
fi

if [[ "${EUID}" -ne 0 ]]; then
  echo "run as root" >&2
  exit 1
fi

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)"
VERSION="$(python3 -c 'import pathlib,sys,tomllib; print(tomllib.loads(pathlib.Path(sys.argv[1]).read_text())["project"]["version"])' "${PROJECT_ROOT}/pyproject.toml")"
RELEASE="/opt/osc-agent/releases/${VERSION}"
if [[ -e "${RELEASE}" ]]; then
  echo "release already exists and is immutable: ${RELEASE}" >&2
  exit 1
fi

install -d -m 0755 /opt/osc-agent /opt/osc-agent/releases
getent group osc-shared >/dev/null || groupadd --system osc-shared
id osc-control >/dev/null 2>&1 || useradd --system --home /var/lib/osc-agent --gid osc-shared osc-control
id osc-worker >/dev/null 2>&1 || useradd --system --home /var/lib/osc-agent --gid osc-shared osc-worker
usermod -aG docker osc-worker
install -d -m 2770 -o osc-control -g osc-shared /var/lib/osc-agent /var/lib/osc-agent/workspaces
install -d -m 0750 -o root -g osc-shared /etc/osc-agent
install -m 0640 -o root -g osc-shared "${CONFIG}" /etc/osc-agent/config.yml
install -d -m 0755 "${RELEASE}" "${RELEASE}/wheels"
python3 -m venv "${RELEASE}/venv"
"${RELEASE}/venv/bin/pip" install --upgrade pip
"${RELEASE}/venv/bin/pip" wheel --wheel-dir "${RELEASE}/wheels" "${PROJECT_ROOT}[bot]"
"${RELEASE}/venv/bin/pip" install --no-index --find-links "${RELEASE}/wheels" \
  "open-source-contribution-agent[bot]==${VERSION}"
ln -sfn "${RELEASE}" /opt/osc-agent/next
mv -Tf /opt/osc-agent/next /opt/osc-agent/current
install -m 0644 "${PROJECT_ROOT}/deploy/ubuntu/osc-agent-bot-control.service" /etc/systemd/system/
install -m 0644 "${PROJECT_ROOT}/deploy/ubuntu/osc-agent-bot-worker.service" /etc/systemd/system/
install -m 0644 "${PROJECT_ROOT}/deploy/ubuntu/osc-agent-bot-cleanup.service" /etc/systemd/system/
install -m 0644 "${PROJECT_ROOT}/deploy/ubuntu/osc-agent-bot-cleanup.timer" /etc/systemd/system/
systemctl daemon-reload
systemctl enable osc-agent-bot-control osc-agent-bot-worker osc-agent-bot-cleanup.timer
echo "Install /etc/osc-agent/bot.env and worker.env with mode 0600, then run /opt/osc-agent/current/venv/bin/osc-agent-bot doctor."
