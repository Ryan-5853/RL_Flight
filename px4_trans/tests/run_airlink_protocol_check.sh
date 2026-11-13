#!/usr/bin/env bash
set -euo pipefail

px4_root="${1:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../px4" && pwd)}"
source_file="$px4_root/src/modules/airlink/AirlinkProtocol.hpp"
test_file="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/airlink_protocol_check.cpp"
module_dir="$(dirname "$source_file")"
session_source="$module_dir/AirlinkSession.cpp"
session_test="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/airlink_session_check.cpp"
landing_test="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/airlink_landing_check.cpp"
build_dir="$(mktemp -d)"
trap 'rm -rf "$build_dir"' EXIT

test -f "$source_file"

"${CXX:-c++}" \
	-std=c++17 \
	-Wall \
	-Wextra \
	-Werror \
	-I"$module_dir" \
	"$test_file" \
	-o "$build_dir/airlink_protocol_check"

"${CXX:-c++}" -std=c++17 -Wall -Wextra -Werror -I"$module_dir" \
	"$session_source" "$(dirname "$test_file")/maintained_arm_switch_check.cpp" \
	-o "$build_dir/maintained_arm_switch_check"
"$build_dir/maintained_arm_switch_check"

"$build_dir/airlink_protocol_check"

"${CXX:-c++}" \
	-std=c++17 \
	-Wall \
	-Wextra \
	-Werror \
	-I"$module_dir" \
	"$session_source" \
	"$session_test" \
	-o "$build_dir/airlink_session_check"

"$build_dir/airlink_session_check"

"${CXX:-c++}" \
	-std=c++17 \
	-Wall \
	-Wextra \
	-Werror \
	-I"$px4_root/src/modules/nn_control" \
	"$landing_test" \
	-o "$build_dir/airlink_landing_check"

"$build_dir/airlink_landing_check"

"${CXX:-c++}" -std=c++17 -Wall -Wextra -Werror -I"$module_dir" \
    "$(dirname "$test_file")/airlink_health_check.cpp" -o "$build_dir/airlink_health_check"
"$build_dir/airlink_health_check"

"${CXX:-c++}" -std=c++17 -Wall -Wextra -Werror -I"$module_dir" \
    "$(dirname "$test_file")/pair_estimate_safety_check.cpp" -o "$build_dir/pair_estimate_safety_check"
"$build_dir/pair_estimate_safety_check"
