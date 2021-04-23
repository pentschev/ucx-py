# Copyright (c) 2019-2021, NVIDIA CORPORATION. All rights reserved.
# See file LICENSE for terms.

# cython: language_level=3

from libc.stdint cimport uint16_t

from .ucx_api_dep cimport *
from .ucx_object cimport UCXObject


cdef class UCXListener(UCXObject):
    cdef:
        ucp_listener_h _handle
        dict cb_data

        public:
            uint16_t port
            str ip
