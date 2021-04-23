# Copyright (c) 2019-2021, NVIDIA CORPORATION. All rights reserved.
# Copyright (c) 2021       UT-Battelle, LLC. All rights reserved.
# See file LICENSE for terms.

# cython: language_level=3

from libc.stdio cimport FILE

from .ucx_api_dep cimport *


cdef FILE * create_text_fd()


cdef unicode decode_text_fd(FILE * text_fd)


cdef void ucx_py_request_reset(void* request)


cdef get_ucx_object(Py_buffer *buffer, int flags,
                    void *ucx_data, Py_ssize_t length, obj)


cdef void assert_ucs_status(ucs_status_t status, str msg_context=*) except *


cdef ucp_config_t * _read_ucx_config(dict user_options) except *


cdef dict ucx_config_to_dict(ucp_config_t *config)
