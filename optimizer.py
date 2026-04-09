import numpy as np
import random
from copy import deepcopy


class WalkForwardOptimizer:
    def __init__(
        self,
        backtester,
        param_space,
        n_trials=100,
        train_size=0.6,
        test_size=0.2,
        step_size=0.2,
        min_folds=3,
        random_seed=42,
    ):
        self.backtester = backtester
        self.param_space = param_space
        self.n_trials = n_trials
        self.train_size = train_size
        self.test_size = test_size
        self.step_size = step_size
        self.min_folds = min_folds
        random.seed(random_seed)

    # =========================
    # PARAM SAMPLING
    # =========================
    def sample_params(self):
        params = {}
        for k, v in self.param_space.items():
            if isinstance(v, list):
                params[k] = random.choice(v)
            elif isinstance(v, tuple):
                params[k] = random.uniform(v[0], v[1])
            else:
                params[k] = v
        return params

    # =========================
    # WALK-FORWARD SPLITS
    # =========================
    def generate_splits(self, data_length):
        train_len = int(data_length * self.train_size)
        test_len = int(data_length * self.test_size)
        step_len = int(data_length * self.step_size)

        print(f"[WF] Train: {train_len} | Test: {test_len} | Step: {step_len}")

        splits = []
        start = 0

        while True:
            train_start = start
            train_end = train_start + train_len
            test_end = train_end + test_len

            if test_end > data_length:
                break

            splits.append((train_start, train_end, train_end, test_end))
            start += step_len

        print(f"[WF] Total splits: {len(splits)}")
        return splits

    # =========================
    # SINGLE CONFIG EVAL
    # =========================
    def evaluate_params(self, data, params):
        splits = self.generate_splits(len(data))

        train_sharpes = []
        test_sharpes = []
        drawdowns = []

        for i, (tr_s, tr_e, te_s, te_e) in enumerate(splits):
            train_data = data.iloc[tr_s:tr_e]
            test_data = data.iloc[te_s:te_e]

            train_res = self.backtester.run(train_data, params)
            test_res = self.backtester.run(test_data, params)

            if train_res is None or test_res is None:
                print(f"[Fold {i}] ❌ Backtest failed")
                continue

            train_sharpes.append(train_res["sharpe"])
            test_sharpes.append(test_res["sharpe"])
            drawdowns.append(test_res["max_drawdown"])

        if len(test_sharpes) < self.min_folds:
            return None

        return {
            "train_sharpes": train_sharpes,
            "test_sharpes": test_sharpes,
            "drawdowns": drawdowns,
        }

    # =========================
    # ROBUSTNESS SCORING
    # =========================
    def compute_score(self, stats):
        train_sharpes = np.array(stats["train_sharpes"])
        test_sharpes = np.array(stats["test_sharpes"])
        drawdowns = np.array(stats["drawdowns"])

        mean_train = np.mean(train_sharpes)
        mean_test = np.mean(test_sharpes)
        std_test = np.std(test_sharpes)

        # Overfit penalty
        overfit_penalty = max(0, mean_train - mean_test)

        # Drawdown penalty
        dd_penalty = np.mean(drawdowns)

        # Robust score
        score = (
            mean_test
            - 0.5 * std_test
            - 1.2 * overfit_penalty
            - 0.3 * dd_penalty
        )

        return {
            "score": score,
            "mean_train": mean_train,
            "mean_test": mean_test,
            "std_test": std_test,
            "overfit": overfit_penalty,
            "avg_dd": dd_penalty,
        }

    # =========================
    # ROBUSTNESS FILTER
    # =========================
    def is_robust(self, metrics):
        if metrics["mean_test"] < 0.1:
            return False

        if metrics["std_test"] > 1.5:
            return False

        if metrics["overfit"] > 1.5:
            return False

        if metrics["avg_dd"] > 0.6:
            return False

        return True

    # =========================
    # MAIN OPTIMIZATION
    # =========================
    def optimize(self, data):
        results = []

        print(f"\n[INFO] Starting optimization with {self.n_trials} trials")
        print(f"[INFO] Data length: {len(data)}\n")

        splits = self.generate_splits(len(data))
        if len(splits) < self.min_folds:
            print("❌ Not enough data for walk-forward")
            return None

        for i in range(self.n_trials):
            params = self.sample_params()

            stats = self.evaluate_params(data, params)
            if stats is None:
                print(f"[{i}] ❌ Not enough valid folds")
                continue

            metrics = self.compute_score(stats)

            print(
                f"[{i}] Train {metrics['mean_train']:.2f} | "
                f"Test {metrics['mean_test']:.2f} | "
                f"Std {metrics['std_test']:.2f} | "
                f"DD {metrics['avg_dd']:.2f}"
            )

            if not self.is_robust(metrics):
                print(f"[{i}] ❌ Rejected")
                continue

            print(f"[{i}] ✅ Accepted")

            results.append({
                "params": deepcopy(params),
                **metrics
            })

        if not results:
            print("\n⚠️ No robust configurations found.")
            return None

        results = sorted(results, key=lambda x: x["score"], reverse=True)

        return results

    # =========================
    # REPORT
    # =========================
    def report(self, results, top_n=5):
        print("\n=== TOP CONFIGURATIONS ===\n")

        for i, res in enumerate(results[:top_n]):
            print(f"Rank {i+1}")
            print(f"Score: {res['score']:.3f}")
            print(f"Train Sharpe: {res['mean_train']:.3f}")
            print(f"Test Sharpe: {res['mean_test']:.3f}")
            print(f"Stability (std): {res['std_test']:.3f}")
            print(f"Overfit: {res['overfit']:.3f}")
            print(f"Avg DD: {res['avg_dd']:.3f}")
            print(f"Params: {res['params']}")
            print("-" * 40)
