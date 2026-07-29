#!/usr/bin/env bash
set -euo pipefail

if [[ "${EUID}" -ne 0 || $# -ne 1 ]]; then
  echo "usage: sudo ./deploy/ubuntu/upgrade.sh /opt/osc-agent/releases/<version>" >&2
  exit 2
fi

RELEASE="$(realpath "$1")"
case "${RELEASE}" in
  /opt/osc-agent/releases/*) ;;
  *) echo "release must be below /opt/osc-agent/releases" >&2; exit 2 ;;
esac
if [[ ! -x "${RELEASE}/venv/bin/osc-agent" ]]; then
  echo "release does not contain an executable venv/bin/osc-agent" >&2
  exit 2
fi

exec 9>/run/lock/osc-agent-upgrade.lock
flock -n 9 || { echo "another upgrade is active" >&2; exit 1; }

# /run is traversable by the unprivileged Nginx worker; /etc/osc-agent is intentionally not.
touch /run/osc-agent-webhook-maintenance
systemctl reload nginx
systemctl stop osc-agent-bot-worker osc-agent-bot-control

ln -sfn "${RELEASE}" /opt/osc-agent/next
mv -Tf /opt/osc-agent/next /opt/osc-agent/current

/opt/osc-agent/current/venv/bin/osc-agent-bot reset-state --confirm
/opt/osc-agent/current/venv/bin/osc-agent-bot schema-check
systemctl start osc-agent-bot-control
systemctl start osc-agent-bot-worker
/opt/osc-agent/current/venv/bin/osc-agent-bot smoke-test
rm -f /run/osc-agent-webhook-maintenance
systemctl reload nginx
