# Copyright (c) 2019-2020, NVIDIA CORPORATION. All rights reserved.
# Copyright (c) 2020       UT-Battelle, LLC. All rights reserved.
# See file LICENSE for terms.

# cython: language_level=3

from .ucx_api_dep cimport *


cdef void _send_callback(void *request, ucs_status_t status)


cdef void _tag_recv_callback(
    void *request, ucs_status_t status, ucp_tag_recv_info_t *info
)


cdef void _stream_recv_callback(
    void *request, ucs_status_t status, size_t length
)


IF CY_UCP_AM_SUPPORTED:
    cdef void _send_nbx_callback(void *request, ucs_status_t status, void *user_data)
