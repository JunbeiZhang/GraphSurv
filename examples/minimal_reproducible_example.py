"""Run the complete GraphSurv workflow on a small synthetic dataset."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from graphsurv.tuning import ensure_basic_columns  # noqa: E402
from graphsurv.workflow import run_graphsurv_cross_validation  # noqa: E402


def generate_synthetic_longitudinal_data(
    seed: int = 7, n_subjects: int = 36
) -> pd.DataFrame:
    """Create irregular longitudinal survival data in GraphSurv CSV format."""
    rng = np.random.default_rng(seed)
    rows = []
    for subject_id in range(n_subjects):
        subject_risk = rng.normal()
        n_visits = int(rng.integers(3, 6))
        visit_times = np.cumsum(rng.uniform(0.25, 1.1, size=n_visits))
        event = int(subject_id % 3 != 0)
        tte = float(visit_times[-1] + rng.uniform(0.5, 1.5) - 0.15 * subject_risk)
        tte = max(tte, float(visit_times[-1] + 0.1))

        for visit_index, time in enumerate(visit_times):
            biomarker_2 = rng.normal(loc=-0.10 * subject_risk, scale=0.45)
            if (subject_id + visit_index) % 4 == 0:
                biomarker_2 = np.nan
            rows.append(
                {
                    "id": subject_id,
                    "time": float(time),
                    "tte": tte,
                    "event": event,
                    "biomarker_1": subject_risk + 0.18 * time + rng.normal(scale=0.25),
                    "biomarker_2": biomarker_2,
                    "biomarker_3": np.sin(time) + 0.25 * subject_risk + rng.normal(scale=0.15),
                }
            )
    return pd.DataFrame(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPOSITORY_ROOT / "examples" / "minimal_reproducible_output",
        help="New or empty directory for generated synthetic outputs.",
    )
    return parser.parse_args()


def require_empty_output_directory(output_dir: Path) -> None:
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(
            f"Output directory must be new or empty so tuning is rerun: {output_dir}"
        )
    output_dir.mkdir(parents=True, exist_ok=True)


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    require_empty_output_directory(output_dir)

    # Step 1: generate synthetic data only; no clinical data are read or written.
    data_path = output_dir / "synthetic_longitudinal_survival_data.csv"
    synthetic_data = ensure_basic_columns(generate_synthetic_longitudinal_data(seed=42))
    synthetic_data.to_csv(data_path, index=False)

    # Steps 2--5: use the release workflow for tuning, fitting, prediction,
    # and IPCW evaluation. The deliberately small settings are only a demo.
    fold_metrics = run_graphsurv_cross_validation(
        data_csv=data_path,
        output_dir=output_dir,
        dataset_name="synthetic",
        n_splits=3,
        seed=42,
        max_epochs=4,
        patience=2,
        K_grid=8,
        batch_size_train=8,
        batch_size_val=16,
        tau_quantile_cut=0.95,
        enable_tuning=True,
        tuning_trials=2,
        tuning_inner_splits=2,
        tuning_epochs=2,
        tuning_patience=1,
    )

    print("\nSynthetic cross-validation metrics")
    print(fold_metrics[["fold", "cindex", "ibs", "iauc"]].to_string(index=False))
    print("\nMean metrics")
    print(fold_metrics[["cindex", "ibs", "iauc"]].mean().to_string())
    print(f"\nOutputs written to: {output_dir}")


if __name__ == "__main__":
    main()
