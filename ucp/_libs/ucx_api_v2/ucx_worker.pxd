# Copyright (c) 2019-2021, NVIDIA CORPORATION. All rights reserved.
# Copyright (c) 2020       UT-Battelle, LLC. All rights reserved.
# See file LICENSE for terms.

# cython: language_level=3

from .ucx_api_dep cimport *
from .ucx_context cimport UCXContext
from .ucx_request cimport UCXRequest
from .ucx_object cimport UCXObject


cdef class UCXWorker(UCXObject):
    cdef:
        ucp_worker_h _handle
        UCXContext _context
        set _inflight_msgs
        IF CY_UCP_AM_SUPPORTED:
            dict _am_recv_pool
            dict _am_recv_wait
            object _am_host_allocator
            object _am_cuda_allocator

    cpdef bint arm(self) except *

    cpdef void request_cancel(self, UCXRequest req) except *

    cpdef ucs_status_t fence(self) except *
