import sys, io; sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
"""
NeuroShift -- Anomaly Detection Example
=========================================
Generates synthetic sensor data (temperature, pressure, vibration),
learns "normal" from 20 examples, then detects anomalies in a mixed stream.

Run:  python -m examples.anomaly_detection   (from project root)
"""

import torch

from neuroshift.hybrid.anomaly_detector import AnomalyDetector

device = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(123)

# ---------------------------------------------------------------------------
# 1. Generate synthetic sensor data
# ---------------------------------------------------------------------------
N_FEATURES = 3  # temperature, pressure, vibration

def make_normal(n):
    """Normal operating conditions."""
    temp     = torch.randn(n, 1) * 2.0 + 70.0    # ~70 C +/- 2
    pressure = torch.randn(n, 1) * 5.0 + 100.0   # ~100 kPa +/- 5
    vib      = torch.randn(n, 1) * 0.3 + 1.0      # ~1.0 mm/s +/- 0.3
    return torch.cat([temp, pressure, vib], dim=1)

def make_anomalous(n):
    """Anomalous readings -- overheating, pressure drops, high vibration."""
    data = []
    per_type = max(1, n // 3)

    # Type A: overheating
    t = torch.randn(per_type, 1) * 3.0 + 95.0
    p = torch.randn(per_type, 1) * 5.0 + 100.0
    v = torch.randn(per_type, 1) * 0.5 + 1.5
    data.append(torch.cat([t, p, v], dim=1))

    # Type B: pressure drop
    t = torch.randn(per_type, 1) * 2.0 + 72.0
    p = torch.randn(per_type, 1) * 3.0 + 60.0
    v = torch.randn(per_type, 1) * 0.4 + 1.2
    data.append(torch.cat([t, p, v], dim=1))

    # Type C: excessive vibration
    remaining = n - 2 * per_type
    t = torch.randn(remaining, 1) * 2.0 + 71.0
    p = torch.randn(remaining, 1) * 5.0 + 98.0
    v = torch.randn(remaining, 1) * 1.0 + 5.0
    data.append(torch.cat([t, p, v], dim=1))

    return torch.cat(data, dim=0)

normal_train = make_normal(20)
normal_test  = make_normal(15)
anomaly_test = make_anomalous(15)

print("=" * 64)
print("  NEUROSHIFT ANOMALY DETECTION EXAMPLE")
print("=" * 64)
print()
print("  Sensor features : temperature (C), pressure (kPa), vibration (mm/s)")
print(f"  Training samples : {normal_train.shape[0]} normal readings")
print(f"  Test samples     : {normal_test.shape[0]} normal + {anomaly_test.shape[0]} anomalous")
print(f"  Device           : {device}")
print()

# ---------------------------------------------------------------------------
# 2. Create detector and learn normal
# ---------------------------------------------------------------------------
detector = AnomalyDetector(feature_dim=N_FEATURES, hdc_dim=10000, device=device)

print("--- Learning normal patterns (instant, no training loop) ---")
detector.learn_normal(normal_train)
print(f"  Learned from {detector.n_normal_seen} examples")
print()

# ---------------------------------------------------------------------------
# 3. Detect anomalies in mixed stream
# ---------------------------------------------------------------------------
print("--- Detection results ---")
print(f"  {'#':>3}  {'Temp':>6}  {'Press':>6}  {'Vib':>6}  {'Score':>6}  {'Conf':>6}  {'Verdict':<10}  {'Truth':<8}")
print("  " + "-" * 62)

tp = fp = tn = fn = 0

# Interleave normal and anomalous for realistic stream
all_data = torch.cat([normal_test, anomaly_test], dim=0)
all_labels = ["normal"] * normal_test.shape[0] + ["ANOMALY"] * anomaly_test.shape[0]

# Shuffle
perm = torch.randperm(all_data.shape[0])
all_data = all_data[perm]
all_labels = [all_labels[i] for i in perm.tolist()]

for i in range(all_data.shape[0]):
    sample = all_data[i]
    truth = all_labels[i]
    result = detector.detect(sample)

    verdict = "ANOMALY" if result["is_anomaly"] else "normal"
    is_correct = verdict == truth

    if truth == "ANOMALY" and verdict == "ANOMALY":
        tp += 1
    elif truth == "normal" and verdict == "ANOMALY":
        fp += 1
    elif truth == "normal" and verdict == "normal":
        tn += 1
    else:
        fn += 1

    mark = "  " if is_correct else "!!"
    print(f"  {i+1:3d}  {sample[0]:6.1f}  {sample[1]:6.1f}  {sample[2]:6.2f}"
          f"  {result['anomaly_score']:6.3f}  {result['confidence']:6.3f}"
          f"  {verdict:<10s}  {truth:<8s} {mark}")

# ---------------------------------------------------------------------------
# 4. Summary
# ---------------------------------------------------------------------------
precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
f1        = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
accuracy  = (tp + tn) / (tp + fp + tn + fn)

print()
print("--- Summary ---")
print(f"  True Positives:  {tp}")
print(f"  False Positives: {fp}")
print(f"  True Negatives:  {tn}")
print(f"  False Negatives: {fn}")
print()
print(f"  Accuracy:  {accuracy * 100:.1f}%")
print(f"  Precision: {precision * 100:.1f}%")
print(f"  Recall:    {recall * 100:.1f}%")
print(f"  F1 Score:  {f1 * 100:.1f}%")
print()

status = detector.get_status()
print("--- Detector status ---")
print(f"  Normal patterns learned : {status['normal_examples_seen']}")
print(f"  Anomalies detected      : {status['anomalies_detected']}")
print(f"  Threshold               : {status['threshold']:.4f}")
print(f"  HDC dimensions          : {status['hdc_dimensions']}")
print(f"  MorphicNet trained      : {status['morphic_trained']}")
print()
print("Done.")
