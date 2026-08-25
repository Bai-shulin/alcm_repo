"""Active-learning Gaussian-process regression pipeline.

Refactored from the original GPR notebook for reproducibility, readability,
and safer GitHub use. The core modeling choices are kept unchanged:
- 50 initial labeled samples
- 50-sample fixed test set
- 20 EI-selected samples per iteration
- LightGBM-based iterative feature screening
- 10 retained features with |Pearson r| < 0.95 redundancy filter
- GPR with Constant * Matern(nu=1.5) + WhiteKernel
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from lightgbm import LGBMRegressor
from matplotlib.colors import LinearSegmentedColormap
from scipy.stats import norm
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler


@dataclass(frozen=True)
class ActiveLearningConfig:
    seed: int = 2
    init_size: int = 50
    test_size: int = 50
    select_per_iter: int = 20
    stop_threshold: float = 0.01
    patience: int = 2
    top_n_features: int = 10
    corr_threshold: float = 0.95
    gpr_restarts: int = 20
    element_col: str = "Dopant_Element"
    # Host chemistry used only for optional dopant-level aggregation.
    # Bi/Te are the defaults for the included case study; change this tuple for
    # another material family without modifying the active-learning loop.
    host_elements: tuple[str, ...] = ("Bi", "Te")


def validate_inputs(X: pd.DataFrame, y: np.ndarray, config: ActiveLearningConfig) -> None:
    if len(X) != len(y):
        raise ValueError(f"Feature/target row mismatch: {len(X)} vs {len(y)}")
    # LightGBM can safely ignore fully missing/partially missing columns during
    # feature ranking. GPR cannot, so selected features are checked separately.
    values = X.to_numpy(dtype=float)
    if np.isinf(values).any():
        raise ValueError("Feature matrix contains infinite values.")
    if not np.isfinite(y).all():
        raise ValueError("Target contains non-finite values.")
    required = config.init_size + config.test_size
    if len(X) <= required:
        raise ValueError(f"Need more than {required} samples, got {len(X)}")


def dynamic_feature_selection(
    X_train_raw: np.ndarray,
    y_train: np.ndarray,
    feature_names: list[str],
    config: ActiveLearningConfig,
):
    df_train = pd.DataFrame(X_train_raw, columns=feature_names)
    model = LGBMRegressor(n_estimators=100, random_state=config.seed, verbose=-1)
    model.fit(df_train, y_train)

    importance = (
        pd.DataFrame({"feature": feature_names, "importance": model.feature_importances_})
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )

    # GPR/StandardScaler require finite selected features. The supplied project
    # matrix contains one fully-NaN legacy column (T1706), so such columns are
    # retained in the input for traceability but excluded from GPR selection.
    finite_features = [
        name for name in feature_names
        if np.isfinite(df_train[name].to_numpy(dtype=float)).all()
    ]
    importance = importance[importance["feature"].isin(finite_features)].reset_index(drop=True)

    correlation = df_train[finite_features].corr().abs()
    selected: list[str] = []
    for feature in importance["feature"]:
        if len(selected) >= config.top_n_features:
            break
        if all(correlation.loc[feature, chosen] < config.corr_threshold for chosen in selected):
            selected.append(feature)

    # Preserve original fallback: fill to top_k even if correlation rule must be relaxed.
    if len(selected) < config.top_n_features:
        for feature in importance["feature"]:
            if feature not in selected:
                selected.append(feature)
            if len(selected) == config.top_n_features:
                break

    idx = [feature_names.index(name) for name in selected]
    raw_importance = importance.set_index("feature").loc[selected, "importance"].to_numpy(dtype=float)
    return idx, selected, raw_importance, model, df_train


def normalize_importance(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    total = values.sum()
    if total <= 0:
        return np.full(len(values), 1.0 / len(values))
    return values / total


def expected_improvement(mu: np.ndarray, sigma: np.ndarray, best_y: float) -> np.ndarray:
    sigma = np.maximum(np.asarray(sigma, dtype=float), 1e-9)
    improvement = np.asarray(mu, dtype=float) - best_y
    z = improvement / sigma
    return improvement * norm.cdf(z) + sigma * norm.pdf(z)


def save_shap_plot(model, train_df, selected_features, iteration: int, save_dir: Path) -> None:
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(train_df)
    selected_idx = [train_df.columns.get_loc(name) for name in selected_features]
    filtered = shap_values[:, selected_idx]
    plot_df = train_df[selected_features]

    plt.figure(figsize=(10, 6))
    shap.summary_plot(filtered, plot_df, feature_names=selected_features, show=False)
    plt.title(f"Iter {iteration} SHAP Summary Plot", fontsize=14)
    plt.tight_layout()
    plt.savefig(save_dir / f"iter_{iteration}_shap_summary.png", dpi=300, bbox_inches="tight")
    plt.close()


def save_corr_heatmap(
    X_train_raw: np.ndarray,
    y_train: np.ndarray,
    all_feature_names: list[str],
    selected_features: list[str],
    iteration: int,
    save_dir: Path,
) -> None:
    import seaborn as sns

    df = pd.DataFrame(X_train_raw, columns=all_feature_names)[selected_features].copy()
    df["Mobility"] = y_train
    corr = df.corr()

    cmap = LinearSegmentedColormap.from_list("custom_bwr", ["#0000FF", "#FFFFFF", "#FF3E3E"], N=200)
    plt.figure(figsize=(11, 9))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap=cmap, vmin=-1, vmax=1, linewidths=0.5, square=True)
    plt.title(f"Pearson correlation analysis (Iter {iteration})", fontsize=14)
    plt.tight_layout()
    plt.savefig(save_dir / f"iter_{iteration}_corr_heatmap.png", dpi=300, bbox_inches="tight")
    plt.close()


def save_parity_and_residual(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    iteration: int,
    r2: float,
    save_dir: Path,
) -> None:
    # Parity plot
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(y_true, y_pred, s=25)
    lo = min(float(np.min(y_true)), float(np.min(y_pred)))
    hi = max(float(np.max(y_true)), float(np.max(y_pred)))
    ax.plot([lo, hi], [lo, hi], linestyle="--", linewidth=1)
    ax.set_xlabel(r"True mobility (cm$^2$ V$^{-1}$ s$^{-1}$)")
    ax.set_ylabel(r"Predicted mobility (cm$^2$ V$^{-1}$ s$^{-1}$)")
    ax.text(0.05, 0.90, rf"$R^2$={r2:.2f}", transform=ax.transAxes)
    ax.text(0.80, 0.08, f"Iter {iteration}", transform=ax.transAxes)
    fig.tight_layout()
    fig.savefig(save_dir / f"iter_{iteration}_parity.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    residual = y_pred - y_true
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(y_pred, residual, s=25)
    ax.axhline(0.0, linestyle="--", linewidth=1)
    ax.set_xlabel(r"Predicted mobility (cm$^2$ V$^{-1}$ s$^{-1}$)")
    ax.set_ylabel("Residual")
    ax.text(0.05, 0.90, rf"$R^2$={r2:.2f}", transform=ax.transAxes)
    ax.text(0.80, 0.08, f"Iter {iteration}", transform=ax.transAxes)
    fig.tight_layout()
    fig.savefig(save_dir / f"iter_{iteration}_residual.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def extract_single_dopant(system_name: str, host_elements=("Bi", "Te")) -> str | None:
    elements = re.findall(r"[A-Z][a-z]?", str(system_name))
    dopants = sorted(set(elements) - set(host_elements))
    return dopants[0] if len(dopants) == 1 else None


def run_active_learning(
    feature_csv: str | Path,
    target_csv: str | Path,
    candidate_csv: str | Path | None = None,
    output_dir: str | Path = "results",
    config: ActiveLearningConfig = ActiveLearningConfig(),
) -> dict[str, object]:
    output_dir = Path(output_dir)
    plot_dir = output_dir / "plots"
    output_dir.mkdir(parents=True, exist_ok=True)
    plot_dir.mkdir(parents=True, exist_ok=True)

    X_df = pd.read_csv(feature_csv)
    y = pd.read_csv(target_csv).iloc[:, 0].to_numpy(dtype=float)
    validate_inputs(X_df, y, config)

    feature_names = X_df.columns.tolist()
    X = X_df.to_numpy(dtype=float)
    rng = np.random.RandomState(config.seed)

    all_indices = np.arange(len(X))
    rng.shuffle(all_indices)
    init_idx = all_indices[: config.init_size]
    remaining_idx = all_indices[config.init_size :]

    X_train_raw = X[init_idx]
    y_train = y[init_idx].copy()

    X_remaining = X[remaining_idx]
    y_remaining = y[remaining_idx]

    shuffled = np.arange(len(X_remaining))
    rng.shuffle(shuffled)
    test_idx = shuffled[: config.test_size]
    pool_idx = shuffled[config.test_size :]

    X_test_raw = X_remaining[test_idx]
    y_test = y_remaining[test_idx]
    X_pool_raw = X_remaining[pool_idx]
    y_pool = y_remaining[pool_idx]

    max_iter = math.ceil(len(X_pool_raw) / config.select_per_iter) + 1
    history = []
    selected_feature_rows = []

    final = {}
    for iteration in range(1, max_iter + 1):
        idx, selected_names, raw_importance, lgb, train_df = dynamic_feature_selection(
            X_train_raw, y_train, feature_names, config
        )
        norm_importance = normalize_importance(raw_importance)

        save_shap_plot(lgb, train_df, selected_names, iteration, plot_dir)
        save_corr_heatmap(X_train_raw, y_train, feature_names, selected_names, iteration, plot_dir)

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train_raw[:, idx])
        X_test = scaler.transform(X_test_raw[:, idx])
        X_pool = scaler.transform(X_pool_raw[:, idx]) if len(X_pool_raw) else np.empty((0, len(idx)))

        kernel = (
            ConstantKernel(1.0, (1e-3, 1e3))
            * Matern(length_scale=np.ones(len(idx)), nu=1.5)
            + WhiteKernel(noise_level=1.0)
        )
        gpr = GaussianProcessRegressor(
            kernel=kernel,
            normalize_y=True,
            n_restarts_optimizer=config.gpr_restarts,
            random_state=config.seed,
        )
        gpr.fit(X_train, y_train)

        y_pred = gpr.predict(X_test)
        r2 = r2_score(y_test, y_pred)
        history.append(
            {
                "Iteration": iteration,
                "Training_Size": len(X_train_raw),
                "Pool_Size": len(X_pool_raw),
                "R2": r2,
            }
        )
        save_parity_and_residual(y_test, y_pred, iteration, r2, plot_dir)

        for rank, (name, imp, weight) in enumerate(zip(selected_names, raw_importance, norm_importance), 1):
            selected_feature_rows.append(
                {
                    "Iteration": iteration,
                    "Rank": rank,
                    "Feature_Name": name,
                    "LGB_Raw_Importance": imp,
                    "Normalized_Weight": weight,
                }
            )

        converged = False
        if iteration >= config.patience:
            recent = [row["R2"] for row in history[-config.patience :]]
            converged = max(recent) - min(recent) < config.stop_threshold

        if converged or len(X_pool_raw) == 0:
            final = {
                "iteration": iteration,
                "r2": r2,
                "selected_names": selected_names,
                "raw_importance": raw_importance,
                "norm_importance": norm_importance,
                "scaler": scaler,
                "gpr": gpr,
                "y_train": y_train.copy(),
            }
            break

        mu, sigma = gpr.predict(X_pool, return_std=True)
        ei = expected_improvement(mu, sigma, np.max(y_train))
        take = min(config.select_per_iter, len(X_pool_raw))
        selected_pool_idx = np.argsort(ei)[::-1][:take]

        X_train_raw = np.vstack([X_train_raw, X_pool_raw[selected_pool_idx]])
        y_train = np.hstack([y_train, y_pool[selected_pool_idx]])

        keep = np.ones(len(X_pool_raw), dtype=bool)
        keep[selected_pool_idx] = False
        X_pool_raw = X_pool_raw[keep]
        y_pool = y_pool[keep]

    if not final:
        raise RuntimeError("Active-learning loop ended without a final model.")

    history_df = pd.DataFrame(history)
    history_df.to_csv(output_dir / "iteration_history.csv", index=False)
    pd.DataFrame(selected_feature_rows).to_csv(output_dir / "selected_features_history.csv", index=False)

    feature_summary = pd.DataFrame(
        {
            "Feature_Name": final["selected_names"],
            "LGB_Raw_Importance": final["raw_importance"],
            "Normalized_Weight": final["norm_importance"],
            "Contribution_Percentage": final["norm_importance"] * 100,
        }
    )
    feature_summary.to_csv(output_dir / "final_feature_summary.csv", index=False)

    candidate_result = None
    dopant_ranking = None
    if candidate_csv is not None:
        candidate_df = pd.read_csv(candidate_csv)
        missing = [name for name in final["selected_names"] if name not in candidate_df.columns]
        if missing:
            raise KeyError(f"Candidate CSV missing selected features: {missing}")

        X_candidate = final["scaler"].transform(candidate_df[final["selected_names"]].to_numpy(dtype=float))
        mu, sigma = final["gpr"].predict(X_candidate, return_std=True)
        ei = expected_improvement(mu, sigma, np.max(final["y_train"]))

        candidate_result = candidate_df.copy()
        candidate_result["Predicted_Mobility"] = mu
        candidate_result["Uncertainty"] = sigma
        candidate_result["EI_Score"] = ei
        candidate_result.to_csv(output_dir / "candidate_all_pred.csv", index=False, encoding="utf-8-sig")

        # Keep composition-level ranking for traceability.
        if config.element_col in candidate_result.columns:
            composition_ranking = (
                candidate_result.groupby(config.element_col, as_index=False)
                .agg(
                    Avg_Pred_Mobility=("Predicted_Mobility", "mean"),
                    Avg_EI=("EI_Score", "mean"),
                    Avg_Uncertainty=("Uncertainty", "mean"),
                    Sample_Count=(config.element_col, "count"),
                )
                .sort_values("Avg_EI", ascending=False)
            )
            composition_ranking.to_csv(output_dir / "composition_ranking.csv", index=False, encoding="utf-8-sig")

            # Optional dopant-element aggregation relative to the configured host chemistry.
            candidate_result["Dopant_Element_Type"] = candidate_result[config.element_col].apply(
                lambda value: extract_single_dopant(value, config.host_elements)
            )
            valid = candidate_result.dropna(subset=["Dopant_Element_Type"])
            dopant_ranking = (
                valid.groupby("Dopant_Element_Type", as_index=False)
                .agg(
                    Average_Predicted_Mobility=("Predicted_Mobility", "mean"),
                    Average_EI=("EI_Score", "mean"),
                    Average_Uncertainty=("Uncertainty", "mean"),
                    Sample_Count=("Dopant_Element_Type", "count"),
                )
                .sort_values("Average_Predicted_Mobility", ascending=False)
            )
            dopant_ranking.to_csv(output_dir / "dopant_element_ranking.csv", index=False, encoding="utf-8-sig")

    summary = pd.DataFrame(
        [{
            "Total_Iterations": final["iteration"],
            "Final_Converged_R2": final["r2"],
            "Initial_Training_Size": config.init_size,
            "Fixed_Test_Size": config.test_size,
            "Selected_Per_Iteration": config.select_per_iter,
            "Top_Features": config.top_n_features,
        }]
    )
    summary.to_csv(output_dir / "run_summary.csv", index=False)

    return {
        "summary": summary,
        "history": history_df,
        "feature_summary": feature_summary,
        "candidate_predictions": candidate_result,
        "dopant_ranking": dopant_ranking,
    }
