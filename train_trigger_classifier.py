"""Train a lightweight STNet-call trigger classifier from KTCal rollouts.

Objective: given per-frame NCC/Kalman diagnostics, predict whether the
current Kalman prediction is worse than a pixel-error threshold (i.e., the
frame is worth spending an STNet call on). The learned score augments the
existing fixed cooldown bypass rules used by KTCalTracker.
"""
import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression

FEATURE_NAMES = [
    'confidence',
    'boundary_peak',
    'epsilon_valid',
    'epsilon_filled',
    'template_age',
]


def parse_args():
    parser = argparse.ArgumentParser('Trigger classifier trainer')
    parser.add_argument('--train_rollout', default='./checkpoint_ktcal_ncc_rollout/train_rollout.npz')
    parser.add_argument('--val_rollout', default='./checkpoint_ktcal_ncc_rollout/val_rollout.npz')
    parser.add_argument('--output', default='./checkpoint_ktcal_trigger/trigger_classifier.npz')
    parser.add_argument('--error_threshold_px', type=float, default=3.0)
    parser.add_argument('--epsilon_fill', type=float, default=0.0,
                        help='Value substituted when epsilon is NaN (no NCC update)')
    parser.add_argument('--penalty', choices=['none', 'l2'], default='l2')
    parser.add_argument('--C', type=float, default=1.0)
    parser.add_argument('--max_iter', type=int, default=200)
    return parser.parse_args()


def build_features(rollout, epsilon_fill):
    confidence = rollout['confidence'].astype(np.float64)
    boundary = rollout['boundary_peak'].astype(np.float64)
    epsilon = rollout['epsilon'].astype(np.float64)
    epsilon_valid = np.isfinite(epsilon).astype(np.float64)
    epsilon_filled = np.where(epsilon_valid > 0.5, epsilon, epsilon_fill)
    template_age = rollout['template_age'].astype(np.float64)
    features = np.stack([
        confidence, boundary, epsilon_valid, epsilon_filled, template_age,
    ], axis=-1)
    return features


def main():
    args = parse_args()
    train_cache = np.load(args.train_rollout, allow_pickle=True)
    val_cache = np.load(args.val_rollout, allow_pickle=True)

    train_features = build_features(train_cache, args.epsilon_fill)
    val_features = build_features(val_cache, args.epsilon_fill)

    train_labels = (train_cache['kalman_error_px'] > args.error_threshold_px).astype(np.int32)
    val_labels = (val_cache['kalman_error_px'] > args.error_threshold_px).astype(np.int32)

    mean = train_features.mean(axis=0)
    std = train_features.std(axis=0)
    std[std < 1e-6] = 1.0
    train_normalized = (train_features - mean) / std
    val_normalized = (val_features - mean) / std

    penalty = None if args.penalty == 'none' else 'l2'
    classifier = LogisticRegression(
        C=args.C, penalty=penalty, solver='lbfgs', max_iter=args.max_iter)
    classifier.fit(train_normalized, train_labels)

    weights = classifier.coef_.astype(np.float64).reshape(-1)
    bias = float(classifier.intercept_[0])

    train_prob = classifier.predict_proba(train_normalized)[:, 1]
    val_prob = classifier.predict_proba(val_normalized)[:, 1]

    def positive_rate_at(threshold, probs):
        return float((probs >= threshold).mean())

    def accuracy_at(threshold, probs, labels):
        return float(((probs >= threshold).astype(int) == labels).mean())

    thresholds = [0.3, 0.4, 0.5, 0.6, 0.7]
    summary = {
        'train_positive_rate': float(train_labels.mean()),
        'val_positive_rate': float(val_labels.mean()),
        'train_scan': [
            {
                'threshold': threshold,
                'positive_rate': positive_rate_at(threshold, train_prob),
                'accuracy': accuracy_at(threshold, train_prob, train_labels),
            } for threshold in thresholds
        ],
        'val_scan': [
            {
                'threshold': threshold,
                'positive_rate': positive_rate_at(threshold, val_prob),
                'accuracy': accuracy_at(threshold, val_prob, val_labels),
            } for threshold in thresholds
        ],
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        output_path,
        weights=weights,
        bias=np.asarray(bias),
        mean=mean.astype(np.float64),
        std=std.astype(np.float64),
        feature_names=np.asarray(FEATURE_NAMES),
        error_threshold_px=np.asarray(args.error_threshold_px, dtype=np.float64),
        epsilon_fill=np.asarray(args.epsilon_fill, dtype=np.float64),
    )
    print(f'saved trigger classifier: {output_path}')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
