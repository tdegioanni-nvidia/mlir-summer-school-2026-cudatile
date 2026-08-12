#!/usr/bin/env bash

install_project() {
    local source_dir="/cuda-tile-project"
    local username
    local project_dir

    printf 'Username: '
    if ! IFS= read -r username; then
        echo "Unable to read the username." >&2
        return 1
    fi

    if [[ -z "$username" || "$username" == "." || "$username" == ".." || "$username" == */* ]]; then
        echo "Enter a non-empty username without slashes." >&2
        return 1
    fi

    if [[ -z "${HOME:-}" || ! -d "$HOME" ]]; then
        echo "HOME is not set to an existing directory." >&2
        return 1
    fi

    project_dir="$HOME/$username"

    if [[ ! -d "$source_dir" ]]; then
        echo "Project directory not found: $source_dir" >&2
        return 1
    fi

    if [[ ! -f "$source_dir/requirements.txt" ]]; then
        echo "Requirements file not found: $source_dir/requirements.txt" >&2
        return 1
    fi

    if ! command -v python3 >/dev/null 2>&1; then
        echo "python3 is required but was not found." >&2
        return 1
    fi

    if [[ -e "$project_dir" ]]; then
        echo "Destination already exists: $project_dir" >&2
        return 1
    fi

    echo "Copying $source_dir to $project_dir..."
    if ! cp -a -- "$source_dir" "$project_dir"; then
        echo "Copy failed. A partial copy may remain at $project_dir." >&2
        return 1
    fi

    if ! cd "$project_dir"; then
        echo "Unable to enter project directory: $project_dir" >&2
        return 1
    fi

    echo "Creating the Python virtual environment..."
    if ! python3 -m venv --clear "$project_dir/venv"; then
        echo "Unable to create the virtual environment." >&2
        return 1
    fi

    echo "Installing Python requirements..."
    if ! "$project_dir/venv/bin/python" -m pip install -r "$project_dir/requirements.txt"; then
        echo "Unable to install the Python requirements." >&2
        return 1
    fi

    # shellcheck disable=SC1091
    source "$project_dir/venv/bin/activate"
    echo "Installation complete: $project_dir"

    if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
        echo "Run 'source $project_dir/restore.sh' to activate the environment in your current shell."
    fi
}

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
    install_project
    exit $?
fi

install_project
return $?
