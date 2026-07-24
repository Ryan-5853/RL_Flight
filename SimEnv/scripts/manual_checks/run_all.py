#!/usr/bin/env python3
"""Run every manual SimEnv dynamics/observation diagnostic."""

from __future__ import annotations

from common import REPO_ROOT
from check_configured_geometry import audit as audit_geometry
from check_control_surface_moments import run as run_surface_checks
from check_observations import run as run_observation_checks
from check_time_integration import run as run_integration_checks


def main() -> None:
    print("=== 1/4 configured geometry ===")
    audit_geometry(REPO_ROOT / "configs" / "example.yaml")
    print("\n=== 2/4 control-surface forces and moments ===")
    run_surface_checks()
    print("\n=== 3/4 multi-step rigid-body integration ===")
    run_integration_checks()
    print("\n=== 4/4 observations and sensors ===")
    run_observation_checks()
    print("\nALL MANUAL CHECKS PASSED")


if __name__ == "__main__":
    main()
