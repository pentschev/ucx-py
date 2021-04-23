# Copyright (c) 2019-2021, NVIDIA CORPORATION. All rights reserved.
# See file LICENSE for terms.

# cython: language_level=3

from cpython.ref cimport PyObject


# Struct used as requests by UCX
cdef struct ucx_py_request:
    bint finished  # Used by downstream projects such as cuML
    unsigned int uid
    PyObject *info
