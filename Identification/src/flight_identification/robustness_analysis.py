from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .robustness_benchmark import (
    CROSS_CELLS,
    SCHEMES,
    SINGLE_DIMS,
    SINGLE_STRENGTHS,
    error_cells,
)


SCHEME_LABELS = {
    "pid": "PID",
    "nominal_lqr": "标称 LQR",
    "oracle_lqr": "真值 LQR",
    "offline_lqr": "离线辨识 LQR(80%)",
    "gru_lqr": "自适应 GRU",
    "e2e_nn": "端到端 NN",
    "aa_nn": "角加速度 NN",
}


def load_metrics(output_dir: Path) -> dict[tuple[str, str], dict[str, Any]]:
    result: dict[tuple[str, str], dict[str, Any]] = {}
    for path in sorted(output_dir.glob("*.json")):
        if path.name in {"summary.json", "matrix.json"}:
            continue
        data = json.loads(path.read_text(encoding="utf-8"))
        result[(data["cell"], data["scheme"])] = data["metrics"]
    return result


def episode_score(metrics: Mapping[str, Any], duration_s: float) -> np.ndarray:
    """Composite score per episode.

    score = 100*converged + 100*safe
            - 100*mean(roll/pitch error) - 60*mean(rate error)
            - 20*saturation_fraction - 2*height_excursion
    """
    arrays = {
        key: np.asarray(metrics[key], dtype=np.float64).reshape(-1)
        for key in (
            "safe",
            "converged",
            "cum_roll_pitch_error_rad_s",
            "cum_rate_error_rad_s",
            "saturation_fraction",
            "height_min_m",
            "height_max_m",
        )
    }
    count = len(arrays["safe"])
    mean_rp = arrays["cum_roll_pitch_error_rad_s"] / duration_s
    mean_rate = arrays["cum_rate_error_rad_s"] / duration_s
    height_excursion = (
        arrays["height_max_m"] - arrays["height_min_m"]
    )
    score = (
        100.0 * arrays["converged"]
        + 100.0 * arrays["safe"]
        - 100.0 * mean_rp
        - 60.0 * mean_rate
        - 20.0 * arrays["saturation_fraction"]
        - 2.0 * height_excursion
    )
    return score


def summarize_cell_scheme(
    metrics: Mapping[str, Any], duration_s: float
) -> dict[str, float]:
    arrays = {
        key: np.asarray(metrics[key], dtype=np.float64).reshape(-1)
        for key in metrics
    }
    scores = episode_score(metrics, duration_s)
    return {
        "n": float(len(arrays["safe"])),
        "composite_score": float(scores.mean()),
        "convergence_rate": float(arrays["converged"].mean()),
        "safety_rate": float(arrays["safe"].mean()),
        "cum_roll_pitch_error_rad_s_mean": float(
            arrays["cum_roll_pitch_error_rad_s"].mean()
        ),
        "cum_rate_error_rad_s_mean": float(
            arrays["cum_rate_error_rad_s"].mean()
        ),
        "cum_yaw_rate_error_rad_s_mean": float(
            arrays["cum_yaw_rate_error_rad_s"].mean()
        ),
        "peak_tilt_rad_mean": float(arrays["peak_tilt_rad"].mean()),
        "peak_rate_rad_s_mean": float(arrays["peak_rate_rad_s"].mean()),
        "final_attitude_error_rad_mean": float(
            arrays["final_attitude_error_rad"].mean()
        ),
        "saturation_fraction_mean": float(
            arrays["saturation_fraction"].mean()
        ),
        "height_excursion_m_mean": float(
            (arrays["height_max_m"] - arrays["height_min_m"]).mean()
        ),
    }


def build_matrix(
    output_dir: Path, duration_s: float
) -> dict[str, Any]:
    metrics = load_metrics(output_dir)
    cells = error_cells()
    matrix: dict[str, Any] = {"cells": [], "schemes": list(SCHEMES)}
    for cell in cells:
        cell_name = cell["name"]
        row: dict[str, Any] = {"cell": cell_name}
        for scheme in SCHEMES:
            key = (cell_name, scheme)
            if key not in metrics:
                continue
            row[scheme] = summarize_cell_scheme(metrics[key], duration_s)
        matrix["cells"].append(row)
    return matrix


def build_report(
    matrix: Mapping[str, Any],
    duration_s: float,
    output_dir: Path,
    figure_dir: Path,
) -> str:
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import font_manager

    cjk_font = Path(
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
    )
    if cjk_font.exists():
        font_manager.fontManager.addfont(str(cjk_font))
        matplotlib.rcParams["font.family"] = [
            "Noto Sans CJK JP",
            "DejaVu Sans",
        ]
    matplotlib.rcParams["axes.unicode_minus"] = False
    import matplotlib.pyplot as plt

    figure_dir.mkdir(parents=True, exist_ok=True)
    cells = matrix["cells"]
    schemes = matrix["schemes"]
    cell_names = [cell["cell"] for cell in cells]
    score = np.full((len(cells), len(schemes)), np.nan)
    for row_index, cell in enumerate(cells):
        for scheme_index, scheme in enumerate(schemes):
            entry = cell.get(scheme)
            if entry is not None:
                score[row_index, scheme_index] = entry["composite_score"]

    # 1. Heatmap: schemes x cells (composite score)
    fig, ax = plt.subplots(figsize=(max(10, 0.28 * len(cell_names)), 6.2))
    im = ax.imshow(score.T, aspect="auto", cmap="RdYlGn", vmin=0.0)
    ax.set_yticks(range(len(schemes)))
    ax.set_yticklabels([SCHEME_LABELS[s] for s in schemes])
    ax.set_xticks(range(len(cell_names)))
    ax.set_xticklabels(cell_names, rotation=90, fontsize=7)
    ax.set_title("Robustness matrix: composite score (higher = better)")
    fig.colorbar(im, ax=ax, label="composite score")
    fig.tight_layout()
    fig.savefig(figure_dir / "heatmap_scheme_cell.png", dpi=150)
    plt.close(fig)

    # 2. Score vs strength per dimension
    dims_present = {
        dim: [1.0, *list(SINGLE_STRENGTHS[dim])]
        for dim in SINGLE_DIMS
    }
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    for dim_index, dim in enumerate(SINGLE_DIMS):
        ax = axes[dim_index // 4][dim_index % 4]
        strengths = [1.0, *SINGLE_STRENGTHS[dim]]
        for scheme_index, scheme in enumerate(schemes):
            values = []
            for strength in strengths:
                cell_name = dim if strength == 1.0 else f"{dim}_x{strength}"
                entry = next(
                    (c for c in cells if c["cell"] == cell_name), None
                )
                if entry is None or entry.get(scheme) is None:
                    values.append(np.nan)
                else:
                    values.append(entry[scheme]["composite_score"])
            ax.plot(strengths, values, marker="o", label=SCHEME_LABELS[scheme])
        ax.set_title(dim)
        ax.set_xlabel("error strength (multiplicative)")
        ax.set_ylabel("composite score")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=6)
    fig.suptitle("Composite score vs single-dimension error strength")
    fig.tight_layout()
    fig.savefig(figure_dir / "strength_curves.png", dpi=150)
    plt.close(fig)

    # 3. Cross-error cells: grouped bars per scheme
    cross_prefixes = tuple(f"{cross['name']}_x" for cross in CROSS_CELLS)
    cross_cells = [
        cell for cell in cells if cell["cell"].startswith(cross_prefixes)
    ]
    cross_names = [
        "mass_x_servo_eff",
        "mass_x_servo_tau",
        "servo_eff_x_noise",
        "motor_eff_x_motor_tau",
        "mass_x_inertia_x_servo_tau",
    ]
    fig, ax = plt.subplots(figsize=(14, 6))
    width = 0.8 / len(schemes)
    x = np.arange(len(cross_names) * 2)
    labels = []
    for index, name in enumerate(cross_names):
        for strength in (0, 1):
            labels.append(f"{name}\nS{strength + 1}")
    for scheme_index, scheme in enumerate(schemes):
        values = []
        for name in cross_names:
            for cell in cells:
                if cell["cell"].startswith(name + "_x"):
                    entry = cell.get(scheme)
                    values.append(
                        np.nan if entry is None else entry["composite_score"]
                    )
        ax.bar(x + scheme_index * width, values, width, label=SCHEME_LABELS[scheme])
    ax.set_xticks(x + width * (len(schemes) - 1) / 2)
    ax.set_xticklabels(labels, fontsize=7)
    ax.set_ylabel("composite score")
    ax.set_title("Cross-dimension error cells (two strengths each)")
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(figure_dir / "cross_error_bars.png", dpi=150)
    plt.close(fig)

    # 4. Component breakdown at baseline and a strong single-dim error
    for cell_name, title in (
        ("baseline", "baseline (no error)"),
        ("servo_eff_x2.0", "servo effectiveness x2.0"),
        ("mass_x2.0", "mass x2.0"),
    ):
        cell = next((c for c in cells if c["cell"] == cell_name), None)
        if cell is None:
            continue
        fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
        names = [
            "convergence_rate",
            "cum_roll_pitch_error_rad_s_mean",
            "cum_rate_error_rad_s_mean",
        ]
        titles = ["convergence rate", "cum |roll/pitch err| (rad·s)", "cum |rate err| (rad·s)"]
        for axis_index, (name, label) in enumerate(zip(names, titles)):
            ax = axes[axis_index]
            values = [
                cell.get(scheme, {}).get(name, np.nan) if cell.get(scheme) else np.nan
                for scheme in schemes
            ]
            ax.bar(
                range(len(schemes)),
                values,
                tick_label=[SCHEME_LABELS[s] for s in schemes],
            )
            ax.set_title(f"{title}: {label}")
            ax.tick_params(axis="x", rotation=30, labelsize=7)
        fig.tight_layout()
        fig.savefig(figure_dir / f"components_{cell_name}.png", dpi=150)
        plt.close(fig)

    # 5. Radar: normalized composite score across dimensions
    try:
        fig, ax = plt.subplots(figsize=(8, 8), subplot_kw={"projection": "polar"})
        dims = SINGLE_DIMS
        angles = np.linspace(0, 2 * np.pi, len(dims), endpoint=False).tolist()
        angles += angles[:1]
        for scheme in schemes:
            values = []
            for dim in dims:
                strength = SINGLE_STRENGTHS[dim][-1]
                cell_name = f"{dim}_x{strength}"
                entry = next((c for c in cells if c["cell"] == cell_name), None)
                baseline_entry = next(
                    (c for c in cells if c["cell"] == "baseline"), None
                )
                base_score = (
                    np.nan
                    if baseline_entry is None or baseline_entry.get(scheme) is None
                    else baseline_entry[scheme]["composite_score"]
                )
                cell_score = (
                    np.nan
                    if entry is None or entry.get(scheme) is None
                    else entry[scheme]["composite_score"]
                )
                values.append(
                    np.nan if np.isnan(base_score) else cell_score / max(base_score, 1.0)
                )
            values += values[:1]
            ax.plot(angles, values, label=SCHEME_LABELS[scheme], linewidth=1.5)
            ax.fill(angles, values, alpha=0.05)
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(dims)
        ax.set_title("Strongest single-dim error score / baseline")
        ax.legend(loc="lower right", fontsize=7)
        fig.tight_layout()
        fig.savefig(figure_dir / "radar.png", dpi=150)
        plt.close(fig)
    except Exception:
        pass

    # Report markdown
    lines = [
        "# 未知样本鲁棒性评测报告",
        "",
        f"- 场景：自稳恢复（初始倾角 15°、角速度 1 rad/s），时长 {duration_s} s，",
        "  虚拟飞手定高（LQR/神经网络方案），模拟传感器噪声；PID 使用原生内嵌高度环。",
        "  - 每个误差单元：8 个未见机体 × 1 初值 = 8 条 episode；",
        "  神经网络方案在代表性子集（baseline + 各维最强强度 + 交叉单元）上补充采样。",
        "- 综合分 = 100·收敛 + 100·安全 − 100·平均滚俯误差 − 60·平均角速度误差",
        "  − 20·饱和占比 − 2·高度漂移幅度（越高越好）。",
        "",
        "## 各方案在各误差单元的综合分",
        "",
        "| 误差单元 | " + " | ".join(SCHEME_LABELS[s] for s in schemes) + " |",
        "| --- | " + " | ".join(["---"] * len(schemes)) + " |",
    ]
    for cell in cells:
        values = []
        for scheme in schemes:
            entry = cell.get(scheme)
            values.append(
                "—" if entry is None else f"{entry['composite_score']:.1f}"
            )
        lines.append(f"| {cell['cell']} | " + " | ".join(values) + " |")
    lines += ["", "## 分量指标（baseline）", ""]
    header = "| 方案 | 收敛率 | 安全率 | 累积滚俯误差(rad·s) | 累积角速度误差(rad·s) | 峰值倾角(rad) | 饱和占比 | 高度漂移(m) |"
    lines.append(header)
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    baseline = next((c for c in cells if c["cell"] == "baseline"), None)
    if baseline is not None:
        for scheme in schemes:
            entry = baseline.get(scheme)
            if entry is None:
                continue
            lines.append(
                f"| {SCHEME_LABELS[scheme]} | {entry['convergence_rate']:.3f} | "
                f"{entry['safety_rate']:.3f} | {entry['cum_roll_pitch_error_rad_s_mean']:.3f} | "
                f"{entry['cum_rate_error_rad_s_mean']:.3f} | {entry['peak_tilt_rad_mean']:.3f} | "
                f"{entry['saturation_fraction_mean']:.3f} | {entry['height_excursion_m_mean']:.2f} |"
            )
    lines += ["", "## 最强单维误差下的分科表现（收敛率 / 累积滚俯误差）", ""]
    lines.append(
        "| 误差单元 | 方案 | 收敛率 | 安全率 | 累积滚俯误差(rad·s) | 累积角速度误差(rad·s) |"
    )
    lines.append("| --- | --- | ---: | ---: | ---: | ---: |")
    for dim in SINGLE_DIMS:
        strength = SINGLE_STRENGTHS[dim][-1]
        cell_name = f"{dim}_x{strength}"
        cell = next((c for c in cells if c["cell"] == cell_name), None)
        if cell is None:
            continue
        for scheme in schemes:
            entry = cell.get(scheme)
            if entry is None:
                continue
            lines.append(
                f"| {cell_name} | {SCHEME_LABELS[scheme]} | "
                f"{entry['convergence_rate']:.3f} | {entry['safety_rate']:.3f} | "
                f"{entry['cum_roll_pitch_error_rad_s_mean']:.3f} | "
                f"{entry['cum_rate_error_rad_s_mean']:.3f} |"
            )
    lines += ["", "## 交叉误差单元综合分", ""]
    lines.append(
        "| 误差单元 | " + " | ".join(SCHEME_LABELS[s] for s in schemes) + " |"
    )
    lines.append("| --- | " + " | ".join(["---"] * len(schemes)) + " |")
    for cell in cells:
        if not cell["cell"].startswith(cross_prefixes):
            continue
        values = []
        for scheme in schemes:
            entry = cell.get(scheme)
            values.append(
                "—" if entry is None else f"{entry['composite_score']:.1f}"
            )
        lines.append(f"| {cell['cell']} | " + " | ".join(values) + " |")
    lines += [
        "",
        "图：heatmap_scheme_cell.png（综合分矩阵）、strength_curves.png（单维误差强度曲线）、",
        "cross_error_bars.png（交叉误差）、components_*.png（分量拆解）、radar.png（最强误差相对保持度）。",
        "",
    ]
    return "\n".join(lines)


def analyze(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = Path(args.output_dir).expanduser().resolve()
    figure_dir = Path(args.figure_dir).expanduser().resolve()
    matrix = build_matrix(output_dir, args.duration_s)
    report = build_report(matrix, args.duration_s, output_dir, figure_dir)
    report_path = Path(args.report).expanduser().resolve()
    report_path.write_text(report, encoding="utf-8")
    matrix_path = output_dir / "matrix.json"
    matrix_path.write_text(json.dumps(matrix, indent=2), encoding="utf-8")
    print(f"report: {report_path}")
    print(f"matrix: {matrix_path}")
    print(f"figures: {figure_dir}")
    return matrix


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Aggregate robustness benchmark results and render figures"
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--figure-dir", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--duration-s", type=float, default=4.0)
    return parser


def main() -> None:
    analyze(build_parser().parse_args())


if __name__ == "__main__":
    main()
