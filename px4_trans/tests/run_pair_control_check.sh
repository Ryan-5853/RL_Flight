#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT
g++ -std=c++17 -O2 -Wall -Wextra -Werror "$ROOT/tests/pair_control_check.cpp" -o "$WORK/check"
"$WORK/check"
