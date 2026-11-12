#!/usr/bin/env bash
# Always create a separate policy environment; never install into simulation.
set -euo pipefail
python_bin=${1:?usage: install.sh /path/to/base/python /new/policy/venv}
venv_dir=${2:?provide a new isolated environment directory}
if [[ -e "${venv_dir}" ]]; then
    echo "Refusing to overwrite an existing environment" >&2
    exit 1
fi
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
"${python_bin}" -m venv "${venv_dir}"
"${venv_dir}/bin/python" -m pip install -r "${SCRIPT_DIR}/requirements.txt"
"${venv_dir}/bin/python" -m pip check
