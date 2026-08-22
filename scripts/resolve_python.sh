#!/usr/bin/env bash

# Select a Python environment that can run this checkout. A copied checkout may
# not contain its original .venv, so the sibling environment is a useful local
# fallback while OPENPI_PYTHON remains available for explicit overrides.
OPENPI_ROOT="${OPENPI_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"

if [[ -n "${OPENPI_PYTHON:-}" ]]; then
    if [[ ! -x "${OPENPI_PYTHON}" ]]; then
        echo "OPENPI_PYTHON is not executable: ${OPENPI_PYTHON}" >&2
        return 1 2>/dev/null || exit 1
    fi
elif [[ -x "${OPENPI_ROOT}/.venv/bin/python" ]]; then
    OPENPI_PYTHON="${OPENPI_ROOT}/.venv/bin/python"
elif [[ -x "${OPENPI_ROOT}/../caption/.venv/bin/python" ]]; then
    OPENPI_PYTHON="${OPENPI_ROOT}/../caption/.venv/bin/python"
elif [[ -x "${OPENPI_ROOT}/../hovapi/.venv/bin/python" ]]; then
    OPENPI_PYTHON="${OPENPI_ROOT}/../hovapi/.venv/bin/python"
elif [[ -x "${OPENPI_ROOT}/../pi05/.venv/bin/python" ]]; then
    OPENPI_PYTHON="${OPENPI_ROOT}/../pi05/.venv/bin/python"
else
    OPENPI_PYTHON="$(command -v python3 || command -v python || true)"
fi

if [[ -z "${OPENPI_PYTHON}" || ! -x "${OPENPI_PYTHON}" ]]; then
    echo "No usable Python interpreter found. Set OPENPI_PYTHON explicitly." >&2
    return 1 2>/dev/null || exit 1
fi

export OPENPI_PYTHON
export PYTHONPATH="${OPENPI_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
