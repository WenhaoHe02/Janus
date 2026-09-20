// SPDX-License-Identifier: Apache-2.0
// Append to the audited Tutti source in an isolated extension, not the live .so.
// Uses its queue structures, cache-volatile CQ polling and fail-closed validator.

#include <c10/cuda/CUDAGuard.h>

__global__ void k_submit_coalesced_v1203(
    tutti_queue_dev qd, uint16_t* tail_state, uint32_t nsid,
    const uint64_t* page_iovas, uint64_t stride, const uint64_t* slbas,
    const int64_t* domains, const int64_t* ids, const int32_t* original_count,
    int capacity, int64_t table_size, uint32_t fixed_bytes,
    const int32_t* lengths, uint64_t* descriptors, const uint64_t* descriptor_iova,
    uint32_t max_bytes, int max_members, bool merge,
    int32_t* command_count, unsigned long long* counters) {
  if (blockIdx.x != 0) return;
  const int n = *original_count;
  CUDA_KERNEL_ASSERT(n >= 0 && n <= capacity);
  if (threadIdx.x == 0) *command_count = 0;
  if (n == 0) return;  // No SQ, CQ or doorbell mutation.

  // Registered descriptor storage is separate from every payload staging row.
  for (int i = threadIdx.x; i < n; i += blockDim.x) {
    const int64_t id = ids[i];
    CUDA_KERNEL_ASSERT(id >= 0 && id < table_size);
    CUDA_KERNEL_ASSERT(domains[id] >= 0);
    const uint32_t bytes = lengths ? static_cast<uint32_t>(lengths[id])
                                   : fixed_bytes;
    CUDA_KERNEL_ASSERT(bytes > 0 && bytes <= 65536 && bytes % 512 == 0);
    CUDA_KERNEL_ASSERT(bytes <= max_bytes);
    CUDA_KERNEL_ASSERT(slbas[id] <= UINT64_MAX - bytes / 512);
    const uint64_t offset = static_cast<uint64_t>(i) * stride;
    descriptors[2 * i] = page_iovas[offset / 65536] + offset % 65536;
    descriptors[2 * i + 1] = bytes;  // Unkeyed Data Block descriptor, type 0.
    // Each producer publishes its descriptors before the eventual SQ doorbell.
    __threadfence_system();
  }
  __syncthreads();
  if (threadIdx.x != 0) return;

  const uint16_t initial_tail = *tail_state;
  int commands = 0;
  unsigned long long total_bytes = 0;
  for (int first = 0; first < n;) {
    const int64_t first_id = ids[first];
    const uint64_t start_lba = slbas[first_id];
    uint32_t bytes = static_cast<uint32_t>(descriptors[2 * first + 1]);
    int end = first + 1;
    while (merge && end < n && end - first < max_members) {
      const int64_t next_id = ids[end];
      const uint32_t next_bytes =
          static_cast<uint32_t>(descriptors[2 * end + 1]);
      if (domains[next_id] != domains[first_id] ||
          start_lba > UINT64_MAX - bytes / 512 ||
          slbas[next_id] != start_lba + bytes / 512 ||
          next_bytes > max_bytes - bytes) {
        break;
      }
      bytes += next_bytes;
      ++end;
    }

    const uint16_t slot_index = static_cast<uint16_t>(
        (static_cast<uint32_t>(initial_tail) + commands) % qd.q_depth);
    nvme_sqe* slot = &qd.sq[slot_index];
    uint4* words = reinterpret_cast<uint4*>(slot);
#pragma unroll
    for (int word = 0; word < 4; ++word) {
      words[word] = make_uint4(0, 0, 0, 0);
    }
    slot->opcode = NVME_OPC_READ;
    slot->flags = NVME_FLAG_PSDT_SGL;
    slot->cid = static_cast<uint16_t>(commands);
    slot->nsid = nsid;
    if (end == first + 1) {
      slot->prp1 = descriptors[2 * first];
      slot->prp2 = bytes;
    } else {
      slot->prp1 = *descriptor_iova + static_cast<uint64_t>(first) * 16;
      // Last Segment SGL descriptor; list entries preserve original GPU IOVAs.
      slot->prp2 = static_cast<uint64_t>((end - first) * 16) |
                   (uint64_t{0x30} << 56);
    }
    slot->cdw10 = static_cast<uint32_t>(start_lba);
    slot->cdw11 = static_cast<uint32_t>(start_lba >> 32);
    slot->cdw12 = bytes / 512 - 1;
    total_bytes += bytes;
    ++commands;
    first = end;
  }
  *command_count = commands;
  atomicAdd(counters + 0, 1ULL);
  atomicAdd(counters + 1, static_cast<unsigned long long>(n));
  atomicAdd(counters + 2, static_cast<unsigned long long>(commands));
  atomicAdd(counters + 3, total_bytes);
  __threadfence_system();
  const uint16_t tail = static_cast<uint16_t>(
      (static_cast<uint32_t>(initial_tail) + commands) % qd.q_depth);
  *tail_state = tail;
  *qd.sq_db = tail;  // Identical one-doorbell policy in both comparison arms.
}

// Submit and poll on one stream. Scratch tensors and registered metadata page
// must be allocated before timing and must outlive this queue's GPU completion.
// domains is the immutable namespace/version/range-domain identity per table row.
void read_coalesced_v1203(
    int64_t sq, int64_t cq, int64_t sq_db, int64_t cq_db,
    int64_t sq_tail, int64_t cq_head, int64_t cq_phase, int64_t timed_out,
    int depth, int qid, int64_t nsid, at::Tensor staging,
    at::Tensor page_iovas, int64_t stride, at::Tensor slbas,
    at::Tensor domains, at::Tensor ids, at::Tensor original_count,
    int fixed_bytes, at::Tensor lengths, at::Tensor statuses,
    at::Tensor command_count, at::Tensor counters, int max_bytes,
    int max_members, bool merge, int64_t max_iters, int64_t stream_ptr,
    bool memory_probe) {
  const auto device = ids.device();
  TORCH_CHECK(ids.is_cuda(), "Selected IDs must be CUDA resident");
  const c10::cuda::CUDAGuard device_guard(device);
  for (const auto& tensor : {page_iovas, slbas, domains, ids, counters}) {
    TORCH_CHECK(tensor.is_cuda() && tensor.device() == device &&
                    tensor.scalar_type() == at::kLong && tensor.dim() == 1 &&
                    tensor.is_contiguous(), "Expected contiguous CUDA int64 vectors");
  }
  for (const auto& tensor : {original_count, lengths, statuses, command_count}) {
    TORCH_CHECK(tensor.is_cuda() && tensor.device() == device &&
                    tensor.scalar_type() == at::kInt && tensor.dim() == 1 &&
                    tensor.is_contiguous(), "Expected contiguous CUDA int32 vectors");
  }
  TORCH_CHECK(staging.is_cuda() && staging.device() == device &&
                  staging.scalar_type() == at::kByte && staging.dim() == 1 &&
                  staging.is_contiguous(), "Expected registered CUDA byte staging");
  const int64_t capacity = ids.numel();
  const int64_t pages = page_iovas.numel();
  TORCH_CHECK(depth >= 2 && depth <= 65535 && capacity > 0 && capacity < depth &&
                  capacity <= 4096 && pages >= 2 && stride > 0 && stride % 65536 == 0 &&
                  capacity <= (pages - 1) * 65536 / stride &&
                  staging.numel() >= pages * 65536,
              "Need a disjoint registered metadata page after all staging rows");
  TORCH_CHECK(original_count.numel() == 1 && command_count.numel() == 1 &&
                  counters.numel() >= 4 && statuses.numel() >= capacity &&
                  slbas.numel() > 0 && domains.numel() == slbas.numel() &&
                  (lengths.numel() == 0 || lengths.numel() == slbas.numel()),
              "Invalid count, status, domain or length-table geometry");
  TORCH_CHECK(max_bytes > 0 && max_bytes <= 33554432 && max_bytes % 512 == 0 &&
                  max_members > 0 && max_members <= 4096 &&
                  (lengths.numel() || (fixed_bytes > 0 && fixed_bytes <= 65536 &&
                                      fixed_bytes % 512 == 0)),
              "Device-verified transfer/SGL limits required");
  TORCH_CHECK(sq && sq_db && sq_tail && nsid > 0 && nsid <= UINT32_MAX &&
                  max_iters > 0 && (memory_probe || (cq && cq_db && cq_head && cq_phase && timed_out)),
              "Invalid queue pointers, namespace or timeout");
  const auto qd = make_qd(sq, cq, sq_db, cq_db, depth, qid);
  const auto stream = reinterpret_cast<cudaStream_t>(stream_ptr);
  const int64_t metadata_offset = (pages - 1) * 65536;
  // Resolve the registered metadata IOVA on device, without CPU count readback.
  const auto* iovas = reinterpret_cast<const uint64_t*>(page_iovas.data_ptr<int64_t>());
  k_submit_coalesced_v1203<<<1, 256, 0, stream>>>(
      qd, reinterpret_cast<uint16_t*>(sq_tail), static_cast<uint32_t>(nsid),
      iovas, static_cast<uint64_t>(stride),
      reinterpret_cast<const uint64_t*>(slbas.data_ptr<int64_t>()),
      domains.data_ptr<int64_t>(), ids.data_ptr<int64_t>(),
      original_count.data_ptr<int32_t>(), static_cast<int>(capacity),
      slbas.numel(), static_cast<uint32_t>(fixed_bytes),
      lengths.numel() ? lengths.data_ptr<int32_t>() : nullptr,
      reinterpret_cast<uint64_t*>(staging.data_ptr<uint8_t>() + metadata_offset),
      iovas + pages - 1, static_cast<uint32_t>(max_bytes), max_members, merge,
      command_count.data_ptr<int32_t>(),
      reinterpret_cast<unsigned long long*>(counters.data_ptr<int64_t>()));
  if (!memory_probe) {
    k_poll_batch<false, true><<<1, 256, 0, stream>>>(
        qd, reinterpret_cast<uint16_t*>(cq_head), reinterpret_cast<uint8_t*>(cq_phase),
        static_cast<int>(capacity), reinterpret_cast<uint32_t*>(statuses.data_ptr<int32_t>()),
        reinterpret_cast<int*>(timed_out), static_cast<uint64_t>(max_iters),
        nullptr, nullptr, command_count.data_ptr<int32_t>());
    k_validate_counted_read<<<1, 256, 0, stream>>>(
        command_count.data_ptr<int32_t>(), static_cast<int>(capacity),
        reinterpret_cast<const uint32_t*>(statuses.data_ptr<int32_t>()),
        reinterpret_cast<const int*>(timed_out));
  }
  TORCH_CHECK(cudaGetLastError() == cudaSuccess, "Coalesced read kernel launch failed");
}
