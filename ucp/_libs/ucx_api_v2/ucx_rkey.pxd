# Copyright (c) 2021       UT-Battelle, LLC. All rights reserved.
# See file LICENSE for terms.

# cython: language_level=3

from .ucx_api_dep cimport *
from .ucx_endpoint cimport UCXEndpoint
from .ucx_object cimport UCXObject


cdef class UCXRkey(UCXObject):
    cdef:
        ucp_rkey_h _handle
        UCXEndpoint ep
