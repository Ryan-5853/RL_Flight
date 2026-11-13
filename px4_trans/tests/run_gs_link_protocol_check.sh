#!/usr/bin/env bash
# Host checks for GS-Link v1 wire constants and CRC.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT

g++ -std=c++17 -O2 -Wall -Wextra -Werror \
	-I"$ROOT/px4/src/modules/gs_link" \
	"$ROOT/tests/gs_link_protocol_check.cpp" \
	-o "$WORK/gs_link_protocol_check"

"$WORK/gs_link_protocol_check"

"${CXX:-c++}" -std=c++17 -O2 -Wall -Wextra -Werror \
    "$ROOT/tests/gs_pair_diagnostics_check.cpp" -o "$WORK/gs_pair_diagnostics_check"
"$WORK/gs_pair_diagnostics_check"

"${CXX:-c++}" -std=c++17 -O2 -Wall -Wextra -Werror \
    "$ROOT/tests/gs_mocap_check.cpp" -o "$WORK/gs_mocap_check"
"$WORK/gs_mocap_check"
