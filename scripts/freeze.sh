#!/usr/bin/env bash
# Snapshot the exact versions that produced your results.
pip freeze > requirements.lock.txt
echo "wrote requirements.lock.txt"
