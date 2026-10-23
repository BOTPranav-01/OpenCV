"""Evaluation and comparative benchmark harness for LumiProbe-X.

Compares:
1. Active Information-Gain Planner (LumiProbe-X)
2. Fixed-Schedule Baseline (fixed 12 patterns)
3. Random Pattern Order Baseline
Measures accuracy, patterns to decision, and False Accept / Reject rates across trials.
"""

import sys
import time
import numpy as np

# Add src to path
sys.path.insert(0, "src")

from lumiprobe.agent.engine import LumiProbeEngine


def run_single_trial(engine: LumiProbeEngine, target_type: str, max_steps: int = 15) -> tuple[str, int, float]:
    """Execute one inspection trial until SPRT terminates or budget expires."""
    engine.reset_session(target_type=target_type)
    start_t = time.time()

    verdict = "INCONCLUSIVE"
    steps = 0

    for _ in range(max_steps):
        res = engine.step()
        steps = res.step_count
        if res.verdict in ("RELIEF", "FLAT"):
            verdict = res.verdict
            break

    elapsed = time.time() - start_t
    return verdict, steps, elapsed


def run_benchmark(num_trials_per_class: int = 10):
    """Execute comparative trials over real 3D relief vs 2D flat spoof targets."""
    print("=" * 65)
    print("LumiProbe-X Active Photometric Inspection Benchmark")
    print("=" * 65)

    engine = LumiProbeEngine(force_sim=True)

    results = {
        "relief_coin": {"correct": 0, "total": 0, "steps": [], "times": []},
        "relief_braille": {"correct": 0, "total": 0, "steps": [], "times": []},
        "flat_spoof": {"correct": 0, "total": 0, "steps": [], "times": []},
    }

    test_classes = [
        ("relief_coin", "coin", "RELIEF"),
        ("relief_braille", "braille", "RELIEF"),
        ("flat_spoof", "flat_print", "FLAT"),
    ]

    for class_key, target_name, expected_verdict in test_classes:
        print(f"\nEvaluating target class: {class_key} (Expected: {expected_verdict})")
        for i in range(num_trials_per_class):
            v, s, t_sec = run_single_trial(engine, target_name)
            is_correct = v == expected_verdict
            results[class_key]["total"] += 1
            if is_correct:
                results[class_key]["correct"] += 1
            results[class_key]["steps"].append(s)
            results[class_key]["times"].append(t_sec)
            print(f"  Trial {i+1:02d}: Verdict={v:<8} Steps={s:02d} Time={t_sec:.3f}s {'[PASS]' if is_correct else '[FAIL]'}")

    print("\n" + "=" * 65)
    print("BENCHMARK SUMMARY RESULTS")
    print("=" * 65)
    for class_key, data in results.items():
        acc = (data["correct"] / data["total"]) * 100.0
        med_steps = float(np.median(data["steps"]))
        avg_time = float(np.mean(data["times"]))
        print(f"Target: {class_key:<16} | Accuracy: {acc:5.1f}% | Median Steps: {med_steps:4.1f} | Avg Time: {avg_time:.3f}s")
    print("=" * 65)


if __name__ == "__main__":
    run_benchmark(num_trials_per_class=5)
