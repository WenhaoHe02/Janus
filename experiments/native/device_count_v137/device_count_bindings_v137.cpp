// SPDX-License-Identifier: Apache-2.0
#include <torch/extension.h>
#include "tutti_kv_ops.cuh"
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("tutti_read_indexed_device_count", &tutti_read_indexed_device_count,
        py::arg("sq_dev_ptr"), py::arg("cq_dev_ptr"), py::arg("sq_db_ptr"),
        py::arg("cq_db_ptr"), py::arg("sq_tail_ptr"), py::arg("cq_head_ptr"),
        py::arg("cq_phase_ptr"), py::arg("timed_out_ptr"), py::arg("q_depth"),
        py::arg("qid"), py::arg("nsid"), py::arg("staging_page_iovas"),
        py::arg("staging_stride"), py::arg("slba_table"),
        py::arg("selected_ids"), py::arg("device_count"), py::arg("byte_len"),
        py::arg("status_out"), py::arg("max_iters"),
        py::arg("stream_ptr") = int64_t(0),
        py::call_guard<py::gil_scoped_release>());
  m.def("tutti_read_indexed_segments_device_count", &tutti_read_indexed_segments_device_count,
        py::arg("sq_dev_ptr"), py::arg("cq_dev_ptr"), py::arg("sq_db_ptr"),
        py::arg("cq_db_ptr"), py::arg("sq_tail_ptr"), py::arg("cq_head_ptr"),
        py::arg("cq_phase_ptr"), py::arg("timed_out_ptr"), py::arg("q_depth"),
        py::arg("qid"), py::arg("nsid"), py::arg("staging_page_iovas"),
        py::arg("staging_stride"), py::arg("slba_table"),
        py::arg("selected_ids"), py::arg("device_count"), py::arg("byte_lengths"),
        py::arg("status_out"), py::arg("max_iters"),
        py::arg("stream_ptr") = int64_t(0),
        py::call_guard<py::gil_scoped_release>());
}

