#!/usr/bin/env python3

# Copyright (c)  2023  Xiaomi Corporation
# Author: Fangjun Kuang

from typing import Dict

import numpy as np
import onnx


def load_cmvn():
    neg_mean = None
    inv_stddev = None

    with open("onnx1/am.mvn") as f:
        for line in f:
            if not line.startswith("<LearnRateCoef>"):
                continue
            t = line.split()[3:-1]

            if neg_mean is None:
                neg_mean = ",".join(t)
            else:
                inv_stddev = ",".join(t)

    return neg_mean, inv_stddev


def load_lfr_params():
    with open("onnx1/config.yaml") as f:
        for line in f:
            if "lfr_m" in line:
                lfr_m = int(line.split()[-1])
            elif "lfr_n" in line:
                lfr_n = int(line.split()[-1])
                break
    lfr_window_size = lfr_m
    lfr_window_shift = lfr_n
    return lfr_window_size, lfr_window_shift

def get_predictor_bias():
    with open("onnx1/config.yaml") as f:
        for line in f:
            if "predictor_bias" in line:
                predictor_bias = int(line.split()[-1])
                break
    return predictor_bias

def get_num_mels():
    with open("onnx1/config.yaml") as f:
        for line in f:
            if "n_mels" in line:
                num_mels = int(line.split()[-1])
                break
    return num_mels


def get_vocab_size():
    with open("onnx1/tokens.txt") as f:
        return len(f.readlines())




def add_meta_data(filename: str, meta_data: Dict[str, str]):
    """Add meta data to an ONNX model. It is changed in-place.

    Args:
      filename:
        Filename of the ONNX model to be changed.
      meta_data:
        Key-value pairs.
    """
    model = onnx.load(filename)
    for key, value in meta_data.items():
        meta = model.metadata_props.add()
        meta.key = key
        meta.value = value

    onnx.save(model, filename)
    print(f"Updated {filename}")


def main():
    lfr_window_size, lfr_window_shift = load_lfr_params()
    neg_mean, inv_stddev = load_cmvn()
    vocab_size = get_vocab_size()
    predictor_bias = get_predictor_bias()
    print (predictor_bias)
    num_mels = get_num_mels()
    print (num_mels)
    meta_data = {
        "lfr_window_size": str(lfr_window_size),
        "lfr_window_shift": str(lfr_window_shift),
        "neg_mean": neg_mean,
        "inv_stddev": inv_stddev,
        "model_type": "paraformer2",
        "version": "1",
        "model_author": "k2-fsa",
        "vocab_size": str(vocab_size),
        "predictor_bias": str(predictor_bias),
        "n_mels": str(num_mels),
        "comment": "paraformer encoder+predictor+decoder"
    }
    add_meta_data("onnx1/encoder.onnx", meta_data)


if __name__ == "__main__":
    main()
