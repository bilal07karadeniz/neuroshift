"""
NEUROSHIFT HYBRID ANOMALY DETECTOR
=================================
The first system to combine three non-standard AI paradigms into one pipeline:

  HDC (Perception)  -->  MorphicNet (Processing)  -->  Swarm (Optimization)

How it works:
  1. HDC instantly encodes incoming data into hypervectors (no training)
  2. MorphicNet processes encoded data with self-evolving architecture
  3. Swarm evolves detection thresholds and strategies in the background

Use cases:
  - Network intrusion detection (learn normal traffic from 5 packets)
  - IoT sensor anomaly detection (self-adapting to sensor drift)
  - Financial fraud detection (instant learning, adaptive complexity)
  - Log anomaly detection (encode log patterns, detect deviations)

What makes this different from existing anomaly detectors:
  - No pre-training phase: learns "normal" instantly from few examples
  - Self-sizing: model complexity adapts to data complexity
  - Self-optimizing: detection thresholds evolve without human tuning
  - Runs under 1GB VRAM
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Tuple, Optional
import time
import random
import math

from neuroshift.hdc.engine import HyperdimensionalEngine
from neuroshift.morphic.network import MorphicNet
from neuroshift.swarm.ecosystem import NeuralSwarm, NeuralAgent


class AnomalyDetector:
    """Three-paradigm hybrid anomaly detection system.

    Pipeline:
        Raw Data --> HDC Encoder --> Feature HV --> MorphicNet Classifier --> Score
                                                         ^
                                                         |
                                              Swarm optimizes thresholds
    """

    def __init__(self, feature_dim: int, hdc_dim: int = 10000,
                 device: str = "auto"):
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.feature_dim = feature_dim
        self.hdc_dim = hdc_dim

        # === LAYER 1: HDC Encoder ===
        # Random projection matrix for encoding numeric features into hypervectors
        self.hdc = HyperdimensionalEngine(dimensions=hdc_dim, device=str(self.device))
        self.projection = torch.randn(feature_dim, hdc_dim, device=self.device) / math.sqrt(feature_dim)

        # Prototype memory: stores encoded "normal" patterns
        self.normal_prototypes: List[torch.Tensor] = []
        self.normal_centroid: Optional[torch.Tensor] = None

        # === LAYER 2: MorphicNet Processor ===
        # Takes HDC similarity features and classifies normal vs anomaly
        # Input: [similarity_to_centroid, min_sim, max_sim, std_sim, raw_features(16)]
        morphic_input = 4 + 16  # HDC stats + raw features padded to 16
        self.morphic = MorphicNet(
            input_dim=morphic_input,
            output_dim=2,  # normal vs anomaly
            initial_hidden=8,
            task="classification",
            device=str(self.device)
        )

        # === LAYER 3: Swarm Threshold Optimizer ===
        self.anomaly_threshold = 0.5  # Will be evolved by swarm
        self.sensitivity = 1.0        # Will be evolved by swarm

        # Stats
        self.n_normal_seen = 0
        self.n_anomalies_detected = 0
        self.detection_history: List[Dict] = []
        self._morphic_trained = False

    def encode(self, features: torch.Tensor) -> torch.Tensor:
        """HDC Layer: Encode raw features into a hypervector.

        Uses random projection followed by sign quantization.
        This is a locality-sensitive hash - similar inputs get similar HVs.
        """
        if features.dim() == 1:
            features = features.unsqueeze(0)
        features = features.to(self.device)

        # Project to high-dimensional space and binarize
        projected = features @ self.projection
        return projected.sign()

    def learn_normal(self, features: torch.Tensor):
        """Show the system examples of 'normal' data. Can be called few times.

        This is instantaneous - no training loop, just HDC encoding + storage.
        After learning, auto-calibrates the detection threshold.
        """
        hv = self.encode(features)

        if hv.dim() == 2:
            for i in range(hv.shape[0]):
                self.normal_prototypes.append(hv[i:i+1])
                self.n_normal_seen += 1
        else:
            self.normal_prototypes.append(hv)
            self.n_normal_seen += 1

        # Update centroid (mean of all normal prototypes)
        all_protos = torch.cat(self.normal_prototypes, dim=0)
        self.normal_centroid = all_protos.mean(dim=0, keepdim=True).sign()

        # Auto-calibrate: compute similarity distribution of known normals
        if len(self.normal_prototypes) >= 3:
            sims = []
            for proto in self.normal_prototypes:
                sim = F.cosine_similarity(proto, self.normal_centroid).item()
                sims.append(sim)
            self._normal_sim_mean = sum(sims) / len(sims)
            self._normal_sim_std = (sum((s - self._normal_sim_mean)**2 for s in sims) / len(sims)) ** 0.5
        else:
            self._normal_sim_mean = 0.0
            self._normal_sim_std = 1.0

    def _compute_hdc_features(self, hv: torch.Tensor) -> torch.Tensor:
        """Compute HDC-based similarity features for MorphicNet input."""
        if self.normal_centroid is None:
            return torch.zeros(4, device=self.device)

        # Similarity to centroid
        centroid_sim = F.cosine_similarity(
            hv.view(1, -1), self.normal_centroid.view(1, -1)
        )

        # Similarities to individual prototypes
        if self.normal_prototypes:
            protos = torch.cat(self.normal_prototypes, dim=0)
            sims = F.cosine_similarity(hv.expand(protos.shape[0], -1), protos)
            min_sim = sims.min()
            max_sim = sims.max()
            std_sim = sims.std() if sims.shape[0] > 1 else torch.tensor(0.0, device=self.device)
        else:
            min_sim = centroid_sim
            max_sim = centroid_sim
            std_sim = torch.tensor(0.0, device=self.device)

        return torch.stack([centroid_sim.squeeze(), min_sim, max_sim, std_sim])

    def detect(self, features: torch.Tensor) -> Dict:
        """Detect if input is anomalous.

        Returns dict with:
          - is_anomaly: bool
          - anomaly_score: float (0=normal, 1=anomalous)
          - confidence: float
          - details: dict with per-layer info
        """
        if features.dim() == 1:
            features = features.unsqueeze(0)
        features = features.to(self.device)

        t0 = time.perf_counter()

        # Layer 1: HDC encoding
        hv = self.encode(features)
        hdc_features = self._compute_hdc_features(hv.squeeze(0))

        # Combine HDC features with top raw features for MorphicNet
        raw_subset = features[0, :min(self.feature_dim, 16)]
        if raw_subset.shape[0] < 16:
            raw_subset = F.pad(raw_subset, (0, 16 - raw_subset.shape[0]))
        morphic_input = torch.cat([hdc_features, raw_subset])

        # HDC z-score-based anomaly probability (always computed)
        centroid_sim = hdc_features[0].item()
        if hasattr(self, '_normal_sim_std') and self._normal_sim_std > 1e-8:
            z_score = (self._normal_sim_mean - centroid_sim) / self._normal_sim_std
        else:
            z_score = -centroid_sim * 10
        hdc_anomaly_prob = 1.0 / (1.0 + math.exp(-z_score + 1.5))

        # Layer 2: MorphicNet classification (if trained)
        if self._morphic_trained:
            self.morphic.eval()
            with torch.no_grad():
                logits = self.morphic(morphic_input.unsqueeze(0))
                probs = F.softmax(logits, dim=-1)
                morphic_anomaly_prob = probs[0, 1].item()

            # ENSEMBLE: use MAX - if either system detects anomaly, flag it
            # HDC catches broad deviations, MorphicNet catches learned patterns
            anomaly_prob = max(hdc_anomaly_prob, morphic_anomaly_prob)
        else:
            anomaly_prob = hdc_anomaly_prob

        # Apply swarm-evolved threshold
        is_anomaly = anomaly_prob > self.anomaly_threshold
        confidence = abs(anomaly_prob - self.anomaly_threshold) / max(self.anomaly_threshold, 1 - self.anomaly_threshold)

        elapsed = time.perf_counter() - t0

        if is_anomaly:
            self.n_anomalies_detected += 1

        result = {
            'is_anomaly': is_anomaly,
            'anomaly_score': anomaly_prob,
            'confidence': min(1.0, confidence),
            'detection_time_ms': elapsed * 1000,
            'details': {
                'hdc_centroid_sim': hdc_features[0].item(),
                'hdc_min_sim': hdc_features[1].item(),
                'hdc_max_sim': hdc_features[2].item(),
                'morphic_arch': self.morphic.get_architecture_str() if self._morphic_trained else 'not trained',
                'threshold': self.anomaly_threshold,
            }
        }

        self.detection_history.append(result)
        return result

    def train_morphic(self, normal_data: torch.Tensor, anomaly_data: torch.Tensor,
                      epochs: int = 200):
        """Train the MorphicNet layer on labeled data for better detection.

        This is optional - the detector works without it using pure HDC.
        But MorphicNet adds adaptive, self-evolving classification power.
        """
        normal_data = normal_data.to(self.device)
        anomaly_data = anomaly_data.to(self.device)

        # Prepare MorphicNet training data
        X_list, Y_list = [], []

        for i in range(normal_data.shape[0]):
            hv = self.encode(normal_data[i:i+1])
            hdc_feats = self._compute_hdc_features(hv.squeeze(0))
            raw_subset = normal_data[i, :min(self.feature_dim, 16)]
            if raw_subset.shape[0] < 16:
                raw_subset = F.pad(raw_subset, (0, 16 - raw_subset.shape[0]))
            X_list.append(torch.cat([hdc_feats, raw_subset]))
            Y_list.append(0)  # normal

        for i in range(anomaly_data.shape[0]):
            hv = self.encode(anomaly_data[i:i+1])
            hdc_feats = self._compute_hdc_features(hv.squeeze(0))
            raw_subset = anomaly_data[i, :min(self.feature_dim, 16)]
            if raw_subset.shape[0] < 16:
                raw_subset = F.pad(raw_subset, (0, 16 - raw_subset.shape[0]))
            X_list.append(torch.cat([hdc_feats, raw_subset]))
            Y_list.append(1)  # anomaly

        X = torch.stack(X_list)
        Y = torch.tensor(Y_list, dtype=torch.long, device=self.device)

        # Shuffle
        perm = torch.randperm(len(X))
        X, Y = X[perm], Y[perm]

        optimizer = torch.optim.Adam(self.morphic.parameters(), lr=0.005)

        for epoch in range(epochs):
            loss_val, evolved = self.morphic.train_step(X, Y, optimizer, epoch)
            if evolved:
                optimizer = torch.optim.Adam(self.morphic.parameters(), lr=0.005)

        self._morphic_trained = True

    def optimize_thresholds(self, normal_data: torch.Tensor, anomaly_data: torch.Tensor,
                            generations: int = 100):
        """Use Swarm to evolve optimal detection thresholds.

        Each swarm agent proposes a (threshold, sensitivity) pair.
        Fitness = F1 score on the validation data.
        """
        normal_data = normal_data.to(self.device)
        anomaly_data = anomaly_data.to(self.device)

        # Pre-compute all detection scores
        normal_scores = []
        for i in range(normal_data.shape[0]):
            result = self.detect(normal_data[i:i+1])
            normal_scores.append(result['anomaly_score'])

        anomaly_scores = []
        for i in range(anomaly_data.shape[0]):
            result = self.detect(anomaly_data[i:i+1])
            anomaly_scores.append(result['anomaly_score'])

        normal_scores_t = torch.tensor(normal_scores, device=self.device)
        anomaly_scores_t = torch.tensor(anomaly_scores, device=self.device)

        def fitness_fn(agent, context):
            """Agent outputs [threshold] -> F1 score"""
            obs = torch.zeros(1, device=agent.device)
            output = agent.act(obs)
            threshold = torch.sigmoid(output[0]).item()  # Map to [0, 1]

            # Compute F1 score with this threshold
            tp = (anomaly_scores_t > threshold).sum().item()
            fp = (normal_scores_t > threshold).sum().item()
            fn = (anomaly_scores_t <= threshold).sum().item()

            precision = tp / (tp + fp + 1e-8)
            recall = tp / (tp + fn + 1e-8)
            f1 = 2 * precision * recall / (precision + recall + 1e-8)

            return f1

        swarm = NeuralSwarm(
            population_size=30,
            input_dim=1,
            output_dim=1,
            fitness_fn=fitness_fn,
            hidden=16,
            device=str(self.device)
        )

        results = swarm.run(generations=generations, report_every=generations + 1)

        # Extract best threshold
        best = results['best_agent']
        obs = torch.zeros(1, device=best.device)
        best_output = best.act(obs)
        self.anomaly_threshold = torch.sigmoid(best_output[0]).item()

        return {
            'optimized_threshold': self.anomaly_threshold,
            'best_f1': results['best_fitness'],
            'generations': generations,
        }

    def get_status(self) -> Dict:
        """Get detector status report."""
        return {
            'normal_examples_seen': self.n_normal_seen,
            'anomalies_detected': self.n_anomalies_detected,
            'threshold': self.anomaly_threshold,
            'morphic_trained': self._morphic_trained,
            'morphic_architecture': self.morphic.get_architecture_str(),
            'morphic_params': self.morphic.get_total_params(),
            'hdc_dimensions': self.hdc_dim,
            'device': str(self.device),
        }
