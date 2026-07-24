#!/usr/bin/env python3
"""固定三科评测脚本；功能与 ``flight-train evaluate`` 完全一致。"""

from __future__ import annotations

import argparse

from flight_train.config import load_experiment_config, override_run_config
from flight_train.evaluation import load_fixed_evaluation_suite, run_fixed_evaluation


def main() -> None:
    parser = argparse.ArgumentParser(description="运行飞行自稳控制器固定三科评测")
    parser.add_argument("--config", required=True, help="训练入口配置，用于重建模型和环境")
    parser.add_argument("--checkpoint", required=True, help="带 .sha256 sidecar 的 checkpoint")
    parser.add_argument("--suite", required=True, help="固定评测科目与评分配置")
    parser.add_argument("--device", help="例如 cuda:0；不填时使用训练配置中的 device")
    parser.add_argument("--output-root", help="覆盖评测输出根目录")
    args = parser.parse_args()

    config = override_run_config(
        load_experiment_config(args.config), device=args.device
    )
    suite = load_fixed_evaluation_suite(args.suite)
    result = run_fixed_evaluation(
        config,
        args.checkpoint,
        suite,
        output_root=args.output_root,
    )
    print(result["evaluation_directory"])


if __name__ == "__main__":
    main()
