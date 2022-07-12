#!/usr/bin/env python
# -*- coding: utf-8 -*-
import random
import pytest
import numpy as np
from tacoma.utils import ResidualVariance, Isomap_max_kneighbors,PODTruncation

rng = np.random.RandomState(42)


X = rng.random_sample(size = (100,111_000))

def test_ResidualVariance():
    res_val = ResidualVariance(X)
    res_val.get_results()
    res_val.plot()


def test_Isomap_max_kneighbors():
    Isomap_max_kneighbors(X)


def test_PODTruncation():
    trunc = PODTruncation(X)
    trunc.plot()

