#!/bin/bash
# Clean runner for macOS/Linux
# Suppresses all Python warnings

export PYTHONWARNINGS="ignore"
python3 -W ignore main.py