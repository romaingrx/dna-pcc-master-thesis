#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author : Romain Graux
@date : 2022 May 11, 18:12:12
@last modified : 2022 May 11, 18:18:39
"""


def train_test_split_ds(ds, validation_split=0.2, test_split=0.0):
    assert 0 <= validation_split <= 1, "validation_split must be between 0 and 1"
    assert 0 <= test_split <= 1, "test_split must be between 0 and 1"
    assert (
        validation_split + test_split <= 1
    ), "validation_split + test_split must be less than 1"

    n = ds.cardinality().numpy()

    validation_size = int(n * validation_split)
    test_size = int(n * test_split)
    train_size = max(n - test_size - validation_size, 0)

    train_ds = ds.take(train_size)
    validation_ds = ds.skip(train_size)

    if test_split > 0:
        test_ds = validation_ds.skip(validation_size)
        validation_ds = validation_ds.take(validation_size)
        return train_ds, validation_ds, test_ds
    return train_ds, validation_ds
