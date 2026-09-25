#!/usr/bin/env bash
set -euo pipefail
clear
robolab-run --help
printf '\nEdit the command, then press Enter:\n'
read -er -i 'robolab-run pi05_spring_optimized AnimalsInBinTask' COMMAND
[[ -n "$COMMAND" ]] && bash -lc "$COMMAND"
printf '\nPress Enter to close.'
read -r
