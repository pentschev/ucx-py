# Copyright (c) 2021       UT-Battelle, LLC. All rights reserved.
# See file LICENSE for terms.

# cython: language_level=3

from .ucx_api_dep cimport *
from .ucx_context cimport UCXContext
from .ucx_object cimport UCXObject


cdef class UCXMemoryHandle(UCXObject):
    """ Python representation for ucp_mem_h type. Users should not instance this class
        directly and instead use either the map or the alloc class methods
    """
    cdef:
        ucp_mem_h _mem_handle
        UCXContext _context
        uint64_t r_address
        size_t _length
