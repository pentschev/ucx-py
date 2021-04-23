# Copyright (c) 2019-2021, NVIDIA CORPORATION. All rights reserved.
# Copyright (c) 2020-2021, UT-Battelle, LLC. All rights reserved.
# See file LICENSE for terms.

# cython: language_level=3

from .ucx_api_dep cimport *


cdef class UCXAddress:
    cdef:
        ucp_address_t *_address
        Py_ssize_t _length
