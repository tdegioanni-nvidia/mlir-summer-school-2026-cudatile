#!/usr/bin/env bash

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
    echo "This script must be sourced so it can update the current shell:" >&2
    echo "  source $0" >&2
    exit 1
fi

restore_script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)" || return 1
restore_activate="$restore_script_dir/venv/bin/activate"

if [[ ! -f "$restore_activate" ]]; then
    echo "Virtual environment not found: $restore_activate" >&2
    unset restore_script_dir restore_activate
    return 1
fi

# shellcheck disable=SC1090
source "$restore_activate"
unset restore_script_dir restore_activate
