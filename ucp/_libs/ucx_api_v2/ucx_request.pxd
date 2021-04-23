# Copyright (c) 2019-2021, NVIDIA CORPORATION. All rights reserved.
# See file LICENSE for terms.

# cython: language_level=3

from .typedefs cimport ucx_py_request
from .ucx_api_dep cimport *


cdef class UCXRequest:
    cdef:
        ucx_py_request *_handle
        unsigned int _uid

    cpdef bint closed(self)

    cpdef void close(self) except *


cdef UCXRequest _handle_status(
    ucs_status_ptr_t status,
    int64_t expected_receive,
    cb_func,
    cb_args,
    cb_kwargs,
    unicode name,
    set inflight_msgs
)
