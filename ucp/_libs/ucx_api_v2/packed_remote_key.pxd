# Copyright (c) 2021       UT-Battelle, LLC. All rights reserved.
# See file LICENSE for terms.

# cython: language_level=3


cdef class PackedRemoteKey:
    cdef:
        void *_key
        Py_ssize_t _length
