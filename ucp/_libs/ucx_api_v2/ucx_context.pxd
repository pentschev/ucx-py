# Copyright (c) 2019-2021, NVIDIA CORPORATION. All rights reserved.
# Copyright (c) 2021       UT-Battelle, LLC. All rights reserved.
# See file LICENSE for terms.

# cython: language_level=3

from .ucx_api_dep cimport *
from .ucx_object cimport UCXObject


cdef class UCXContext(UCXObject):
    cdef:
        ucp_context_h _handle
        dict _config
        tuple _feature_flags

        readonly:
            bint cuda_support

    cpdef dict get_config(self)
