#!/bin/bash

# PIP install packages
if [ -f "requirements.txt" ]; then
    pip install -r requirements.txt
fi

# Set the workspace directory as a Git safe directory
git config --global --add safe.directory /workspaces/opik-poc-1

echo "devcontainer-setup.sh complete."
