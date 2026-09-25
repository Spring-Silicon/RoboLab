#!/usr/bin/env bash
set -euo pipefail
clear
RUN="$HOME/.local/bin/robolab-run"
"$RUN" --help
printf '\nEdit the command, then press Enter:\n'
read -er -i "$RUN pi05_spring_optimized AnimalsInBinTask" COMMAND
if [[ -n "$COMMAND" ]]; then
  set +e
  bash -lc "$COMMAND"
  STATUS=$?
  set -e
  printf '\nCommand finished with status %d.\n' "$STATUS"
fi
printf '\nPress Enter to close.'
read -r
