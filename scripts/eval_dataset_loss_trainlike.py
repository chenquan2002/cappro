#!/usr/bin/env python3
"""Evaluate support-caption loss on a replayed approximation of the training stream."""

import argparse

import eval_dataset_loss_common as common


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    common.add_common_args(parser, mode="trainlike")
    parser.add_argument("--replay-file", default=None)
    parser.add_argument("--force-remake-replay", action="store_true")
    parser.add_argument("--train-steps", type=int, default=None)
    parser.add_argument("--train-batch-size", type=int, default=96)
    parser.add_argument("--support-rounds-per-cycle", type=int, default=20)
    parser.add_argument("--replay-seed", type=int, default=None)
    parser.add_argument("--eval-samples", type=int, default=35_750)
    parser.add_argument("--replay-sample-method", choices=["linspace", "random"], default="linspace")
    parser.add_argument("--sample-seed", type=int, default=0)
    return parser.parse_args()


if __name__ == "__main__":
    common.run_mode(parse_args(), mode="trainlike")
