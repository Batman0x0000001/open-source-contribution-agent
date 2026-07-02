from __future__ import annotations

import os
import sys

from osc_agent.tools.shell import MAX_OUTPUT_CHARS, run_bash


def test_run_bash_uses_repo_as_working_directory(tmp_path):
    command = f'{sys.executable} -c "import os; print(os.getcwd())"'

    output = run_bash(command, repo_root=tmp_path)

    assert os.path.normcase(output) == os.path.normcase(str(tmp_path))


def test_run_bash_truncates_large_output(tmp_path):
    command = f'{sys.executable} -c "print(\'x\' * 60000)"'

    output = run_bash(command, repo_root=tmp_path)

    assert len(output) == MAX_OUTPUT_CHARS


def test_run_bash_reports_timeout(tmp_path):
    command = f'{sys.executable} -c "import time; time.sleep(2)"'

    output = run_bash(command, repo_root=tmp_path, timeout_seconds=0.1)

    assert "timed out" in output
