# Copyright (c) 2019-2021, NVIDIA CORPORATION. All rights reserved.
# See file LICENSE for terms.

# cython: language_level=3


cdef class UCXObject:
    cdef:
        object __weakref__
        object _finalizer
        list _children

    cpdef void close(self) except *

    cpdef void add_child(self, child) except *
