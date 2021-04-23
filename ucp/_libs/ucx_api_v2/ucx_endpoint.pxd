# Copyright (c) 2019-2021, NVIDIA CORPORATION. All rights reserved.
# Copyright (c) 2020-2021, UT-Battelle, LLC. All rights reserved.
# See file LICENSE for terms.

# cython: language_level=3

from .ucx_api_dep cimport *
from .ucx_object cimport UCXObject
from .ucx_worker cimport UCXWorker


cdef class UCXEndpoint(UCXObject):
    cdef:
        ucp_ep_h _handle
        set _inflight_msgs

        readonly:
            UCXWorker worker
