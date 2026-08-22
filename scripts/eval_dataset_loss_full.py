#!/usr/bin/env python3
"""Evaluate support-caption loss on every frame from the selected episodes."""

import argparse

import eval_dataset_loss_common as common


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    common.add_common_args(parser, mode="full")
    return parser.parse_args()


if __name__ == "__main__":
    common.run_mode(parse_args(), mode="full")
