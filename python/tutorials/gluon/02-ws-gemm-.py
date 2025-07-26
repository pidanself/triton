import torch
import triton
import triton.language as tl
import pytest

from triton.language.core import _aggregate as aggregate

from triton.experimental import gluon
from triton.experimental.gluon import language as gl
from triton.experimental.gluon.nvidia.hopper import TensorDescriptor
from triton.experimental.gluon.language.nvidia.hopper import fence_async_shared
from triton.experimental.gluon.language.nvidia.blackwell import (
    TensorMemoryLayout,
    allocate_tensor_memory,
    tensor_memory_descriptor,
    tma,
    mbarrier,
    tcgen05_mma as _tcgen05_mma_impl,
    tcgen05_commit,
)

# ===-----------------------------------------------------------------------===#
# Layout Utilities
# ===-----------------------------------------------------------------------===#


@tl.constexpr_function
def get_tmem_32x32b_reg_layout(instr_shape, shape, num_warps):
    assert len(shape) == 2, "expected a 2D tensor"
    assert num_warps in [4, 8], "expected 4 or 8 warps"
    M, N, _ = instr_shape

    blocks_per_tile = [shape[0] // M, shape[1] // N]
    num_blocks = blocks_per_tile[0] * blocks_per_tile[1]

    num_warp_groups = num_warps // 4
    if M == 64:
        threads_per_warp = [16, 2]
        if num_blocks == 1:
            size_per_thread = [1, N // (num_warp_groups * 2)]
            warps_per_cta = [4, num_warp_groups]
        else:
            size_per_thread = [1, N // 2]
            warps_per_cta = [4 * min(blocks_per_tile[0], num_warp_groups)]
            warps_per_cta.append(triton.cdiv(num_warp_groups, warps_per_cta[0] // 4))
    else:
        if shape[0] > 128:
            size_per_thread = [1, N]
            threads_per_warp = [32, 1]
            warps_per_cta = [4 * num_warp_groups, 1]
        else:
            size_per_thread = [1, N // num_warp_groups]
            threads_per_warp = [32, 1]
            warps_per_cta = [4, num_warp_groups]
    return gl.BlockedLayout(
        size_per_thread=size_per_thread,
        threads_per_warp=threads_per_warp,
        warps_per_cta=warps_per_cta,
        order=[0, 1],
    )


@tl.constexpr_function
def get_mma_instr_shape(shape, element_ty):
    m = 128 if shape[0] >= 128 else 64
    n = 256 if shape[1] >= 256 else shape[1]
    k = 256 // element_ty.primitive_bitwidth
    return (m, n, k)


@tl.constexpr_function
def get_nvmma_layout(shape, element_ty, order=[1, 0], fp4_padded=False):
    packing_factor = 2 if fp4_padded else 1

    contig_dim_size = shape[order[0]] * packing_factor * element_ty.primitive_bitwidth // 8
    if contig_dim_size >= 128 and contig_dim_size % 128 == 0:
        swizzle_byte_width = 128
    elif contig_dim_size >= 64 and contig_dim_size % 64 == 0:
        swizzle_byte_width = 64
    elif contig_dim_size >= 32 and contig_dim_size % 32 == 0:
        swizzle_byte_width = 32
    else:
        swizzle_byte_width = 0

    flatten_outer_dim = 1
    for i in range(1, len(shape)):
        flatten_outer_dim *= shape[order[i]]
    if len(shape) < 2 or flatten_outer_dim < 8:
        swizzle_byte_width = 0
    transposed = order[0] == 0

    return gl.NVMMASharedLayout(
        swizzle_byte_width=swizzle_byte_width,
        element_bitwidth=element_ty.primitive_bitwidth,
        rank=len(shape),
        transposed=transposed,
        fp4_padded=fp4_padded,
    )


@tl.constexpr_function
def get_mma_reg_layout(shape, num_warps, dtype=gl.float32):
    instr_shape = get_mma_instr_shape(shape, dtype)
    return get_tmem_32x32b_reg_layout(instr_shape, shape, num_warps)


# ===-----------------------------------------------------------------------===#
# Data Abstractions
# ===-----------------------------------------------------------------------===#

# Channel abstraction for managing memory buffers
def Channel(T, alloc_fn):

    @aggregate
    class ChannelType:
        mem: T
        ready_bars: gl.shared_memory_descriptor
        empty_bars: gl.shared_memory_descriptor
        num_buffers: gl.constexpr
        num_consumers: gl.constexpr

        def __init__(self, mem, ready_bars, empty_bars, num_buffers, num_consumers):
            self.mem = mem
            self.ready_bars = ready_bars
            self.empty_bars = empty_bars
            self.num_buffers = gl.constexpr(num_buffers)
            self.num_consumers = gl.constexpr(num_consumers)

        @gluon.jit
        def alloc(shape: gl.constexpr, dtype: gl.constexpr, layout: gl.constexpr, num_buffers: gl.constexpr,
                  num_consumers: gl.constexpr = 1):
            mem = alloc_fn(dtype, [num_buffers] + shape, layout)
            ready_bars = gl.allocate_shared_memory(gl.int64, [num_buffers, 1], mbarrier.MBarrierLayout())
            empty_bars = gl.allocate_shared_memory(gl.int64, [num_buffers, 1], mbarrier.MBarrierLayout())
            for i in tl.static_range(num_buffers):
                mbarrier.init(ready_bars.index(i), count=1)
                mbarrier.init(empty_bars.index(i), count=num_consumers)
                mbarrier.arrive(empty_bars.index(i), count=num_consumers)
            return ChannelType(mem, ready_bars, empty_bars, num_buffers, num_consumers)

        @gluon.jit
        def increment(self, index, phase):
            if self.num_buffers == 1:
                return gl.to_tensor(0), phase ^ 1
            next_index = index + 1
            rollover = next_index == self.num_buffers
            index = gl.where(rollover, 0, next_index)
            phase = gl.where(rollover, phase ^ 1, phase)
            return index, phase

        @gluon.jit
        def acquire_producer(self, index, phase):
            mem = self.mem.index(index)
            ready_bar = self.ready_bars.index(index)
            empty_bar = self.empty_bars.index(index)

            mbarrier.wait(empty_bar, phase)
            return mem, ready_bar

        @gluon.jit
        def acquire_consumer(self, index, phase):
            mem = self.mem.index(index)
            ready_bar = self.ready_bars.index(index)
            empty_bar = self.empty_bars.index(index)

            mbarrier.wait(ready_bar, phase)
            return mem, empty_bar

        @gluon.jit
        def create_producer(self):
            return Producer(self, gl.to_tensor(0), gl.to_tensor(0))

        @gluon.jit
        def create_consumer(self):
            return Consumer(self, gl.to_tensor(0), gl.to_tensor(0))

        @gluon.jit
        def release(self):
            if isinstance(self.mem, gl.shared_memory_descriptor):
                self.mem._keep_alive()
            for i in tl.static_range(self.num_buffers):
                mbarrier.invalidate(self.ready_bars.index(i))
                mbarrier.invalidate(self.empty_bars.index(i))

    @aggregate
    class Producer:
        channel: ChannelType
        phase: gl.tensor
        index: gl.tensor

        def __init__(self, channel, phase, index):
            self.channel = channel
            self.phase = phase
            self.index = index

        @gluon.jit
        def acquire(self):
            mem, ready_bar = self.channel.acquire_producer(self.index, self.phase)
            self.index, self.phase = self.channel.increment(self.index, self.phase)
            return mem, ready_bar, self

    @aggregate
    class Consumer:
        channel: ChannelType
        phase: gl.tensor
        index: gl.tensor

        def __init__(self, channel, phase, index):
            self.channel = channel
            self.phase = phase
            self.index = index

        @gluon.jit
        def acquire(self):
            mem, empty_bar = self.channel.acquire_consumer(self.index, self.phase)
            self.index, self.phase = self.channel.increment(self.index, self.phase)
            return mem, empty_bar, self

    return ChannelType, Producer, Consumer


SharedMemoryChannel, SharedMemoryProducer, SharedMemoryConsumer = Channel(gl.shared_memory_descriptor,
                                                                          gl.allocate_shared_memory)
TensorMemoryChannel, TensorMemoryProducer, TensorMemoryConsumer = Channel(tensor_memory_descriptor,
                                                                          allocate_tensor_memory)


@gluon.jit
def get_desc_channel(desc, num_buffers: gl.constexpr, num_consumers: gl.constexpr = 1):
    shape: gl.constexpr = desc.block_type.shape
    layout: gl.constexpr = desc.layout
    return SharedMemoryChannel.alloc(shape, desc.dtype, layout, num_buffers, num_consumers)


@gluon.jit
def issue_async_tma_load(smem, bar, desc, offset):
    mbarrier.expect(bar, desc.block_type.nbytes)
    tma.async_copy_global_to_shared(desc, offset, bar, smem)





@gluon.jit
def tcgen05_mma(a, b, d, use_acc, mbarriers):
    _tcgen05_mma_impl(a, b, d, use_acc=use_acc, mbarriers=mbarriers, mbarrier_preds=[True] * len(mbarriers))


# ===-----------------------------------------------------------------------===#
# Matmul Configuration
# ===-----------------------------------------------------------------------===#


@aggregate
class MatmulConfig:
    M: gl.tensor
    N: gl.tensor
    K: gl.tensor

    BLOCK_M: gl.constexpr
    BLOCK_N: gl.constexpr
    BLOCK_K: gl.constexpr
    GROUP_SIZE_M: gl.constexpr
    NUM_SMS: gl.constexpr
    dtype: gl.constexpr
    num_warps: gl.constexpr

    a_shape: gl.constexpr
    b_shape: gl.constexpr
    c_shape: gl.constexpr

    a_tmem_layout: gl.constexpr
    b_tmem_layout: gl.constexpr
    c_tmem_layout: gl.constexpr

    a_layout: gl.constexpr
    b_layout: gl.constexpr
    c_layout: gl.constexpr

    def __init__(self, M, N, K, BLOCK_M, BLOCK_N, BLOCK_K, GROUP_SIZE_M, NUM_SMS, dtype, num_warps):
        self.M = M
        self.N = N
        self.K = K

        self.BLOCK_M = gl.constexpr(BLOCK_M)
        self.BLOCK_N = gl.constexpr(BLOCK_N)
        self.BLOCK_K = gl.constexpr(BLOCK_K)
        self.GROUP_SIZE_M = gl.constexpr(GROUP_SIZE_M)
        self.NUM_SMS = gl.constexpr(NUM_SMS)
        self.dtype = gl.constexpr(dtype)
        self.num_warps = gl.constexpr(num_warps)

        # Shape definitions for matrix blocks
        self.a_shape = gl.constexpr([self.BLOCK_M, self.BLOCK_K])
        self.b_shape = gl.constexpr([self.BLOCK_N, self.BLOCK_K])  # B is [N, K] for standard matmul
        self.c_shape = gl.constexpr([self.BLOCK_M, self.BLOCK_N])

        a_instr_shape = get_mma_instr_shape(self.a_shape, self.dtype)
        b_instr_shape = get_mma_instr_shape(self.b_shape, self.dtype)
        c_instr_shape = get_mma_instr_shape(self.c_shape, self.dtype)
        
        self.a_tmem_layout = gl.constexpr(TensorMemoryLayout((a_instr_shape[0], a_instr_shape[1]), unpacked=True))
        self.b_tmem_layout = gl.constexpr(TensorMemoryLayout((b_instr_shape[0], b_instr_shape[1]), unpacked=True))
        self.c_tmem_layout = gl.constexpr(TensorMemoryLayout((c_instr_shape[0], c_instr_shape[1]), unpacked=True))

        self.a_layout = gl.constexpr(get_tmem_32x32b_reg_layout(a_instr_shape, self.a_shape, self.num_warps))
        self.b_layout = gl.constexpr(get_tmem_32x32b_reg_layout(b_instr_shape, self.b_shape, self.num_warps))
        self.c_layout = gl.constexpr(get_tmem_32x32b_reg_layout(c_instr_shape, self.c_shape, self.num_warps))

    @gluon.jit
    def get_program(self, pid_m, pid_n):
        start_m = pid_m * self.BLOCK_M
        start_n = pid_n * self.BLOCK_N
        return MatmulProgram(self, start_m, start_n)


@aggregate
class MatmulProgram:
    config: MatmulConfig
    start_m: gl.tensor
    start_n: gl.tensor

    def __init__(self, config, start_m, start_n):
        self.config = config
        self.start_m = start_m
        self.start_n = start_n


# ===-----------------------------------------------------------------------===#
# Matmul Kernel
# ===-----------------------------------------------------------------------===#


@gluon.jit
def _matmul_load(config, chnls, descs):
    a_chnl, b_chnl, c_chnl, epi_chnl = chnls
    desc_a, desc_b, desc_c = descs

    a_producer = a_chnl.create_producer()
    b_producer = b_chnl.create_producer()

    # Process all programs assigned to this SM
    scheduler = ProgramScheduler.create(config)
    for pid in range(scheduler.start_pid, scheduler.num_tiles, config.NUM_SMS):
        prog = scheduler.get_program(pid)
        k_tiles = gl.cdiv(config.K, config.BLOCK_K)

        for k in range(k_tiles):
            a_smem, a_bar, a_producer = a_producer.acquire()
            issue_async_tma_load(a_smem, a_bar, desc_a, [prog.start_m, k * config.BLOCK_K])

            b_smem, b_bar, b_producer = b_producer.acquire()
            issue_async_tma_load(b_smem, b_bar, desc_b, [prog.start_n, k * config.BLOCK_K])


@gluon.jit
def _matmul_mma(config, chnls, descs):
    a_chnl, b_chnl, c_chnl, epi_chnl = chnls
    desc_a, desc_b, desc_c = descs

    a_consumer = a_chnl.create_consumer()
    b_consumer = b_chnl.create_consumer()
    c_producer = c_chnl.create_producer()

    # Process all programs assigned to this SM
    scheduler = ProgramScheduler.create(config)
    for pid in range(scheduler.start_pid, scheduler.num_tiles, config.NUM_SMS):
        prog = scheduler.get_program(pid)
        k_tiles = gl.cdiv(config.K, config.BLOCK_K)

        # Initialize accumulator in tensor memory
        c_tmem, c_bar, c_producer = c_producer.acquire()
        
        for k in range(k_tiles):
            a_smem, a_bar, a_consumer = a_consumer.acquire()
            b_smem, b_bar, b_consumer = b_consumer.acquire()

            # Perform matrix multiplication using tcgen05_mma
            # Transpose B for efficient MMA access (B.T for A @ B.T)
            tcgen05_mma(a_smem, b_smem.permute((1, 0)), c_tmem, use_acc=(k > 0), mbarriers=[a_bar, b_bar])
        
        # Commit the barriers after all MMA operations
        tcgen05_commit(a_bar)
        tcgen05_commit(b_bar)


@gluon.jit
def _matmul_epilogue(config, chnls, descs):
    a_chnl, b_chnl, c_chnl, epi_chnl = chnls
    desc_a, desc_b, desc_c = descs

    c_consumer = c_chnl.create_consumer()
    epi_producer = epi_chnl.create_producer()

    # Process all programs assigned to this SM
    scheduler = ProgramScheduler.create(config)
    for pid in range(scheduler.start_pid, scheduler.num_tiles, config.NUM_SMS):
        prog = scheduler.get_program(pid)

        # Get the computed result from tensor memory
        c_tmem, c_bar, c_consumer = c_consumer.acquire()
        
        # Get a shared memory buffer for epilogue
        c_smem, epi_bar, epi_producer = epi_producer.acquire()
        
        # Convert to output dtype and store to shared memory
        c = c_tmem.load(config.c_layout)
        c_smem.store(c.to(config.dtype))
        
        # Use fence to ensure shared memory writes are complete
        fence_async_shared()
        
        # TMA store to global memory (all warps in epilogue can participate)
        tma.async_copy_shared_to_global(desc_c, [prog.start_m, prog.start_n], c_smem)
        
        # Wait for store to complete
        tma.store_wait(0)
        
        # Signal completion
        mbarrier.arrive(c_bar, count=1)
        mbarrier.arrive(epi_bar, count=1)


@aggregate
class ProgramScheduler:
    config: MatmulConfig
    start_pid: gl.tensor
    num_pid_m: gl.tensor
    num_pid_n: gl.tensor
    num_pid_in_group: gl.tensor
    num_tiles: gl.tensor

    def __init__(self, config, start_pid, num_pid_m, num_pid_n, num_pid_in_group, num_tiles):
        self.config = config
        self.start_pid = start_pid
        self.num_pid_m = num_pid_m
        self.num_pid_n = num_pid_n
        self.num_pid_in_group = num_pid_in_group
        self.num_tiles = num_tiles

    @gluon.jit
    def create(config):
        start_pid = gl.program_id(0)
        num_pid_m = gl.cdiv(config.M, config.BLOCK_M)
        num_pid_n = gl.cdiv(config.N, config.BLOCK_N)
        num_pid_in_group = config.GROUP_SIZE_M * num_pid_n
        num_tiles = num_pid_m * num_pid_n
        return ProgramScheduler(config, start_pid, num_pid_m, num_pid_n, num_pid_in_group, num_tiles)

    @gluon.jit
    def get_program(self, tile_id):
        group_id = tile_id // self.num_pid_in_group
        first_pid_m = group_id * self.config.GROUP_SIZE_M
        group_size_m = min(self.num_pid_m - first_pid_m, self.config.GROUP_SIZE_M)
        pid_m = first_pid_m + (tile_id % group_size_m)
        pid_n = (tile_id % self.num_pid_in_group) // group_size_m
        return self.config.get_program(pid_m, pid_n)


def matmul_repr(specialization):
    return "gluon_matmul"


@gluon.jit(do_not_specialize=["M", "N", "K"], repr=matmul_repr)
def matmul_kernel(desc_a, desc_b, desc_c, M, N, K, BLOCK_M: gl.constexpr, BLOCK_N: gl.constexpr, 
                  BLOCK_K: gl.constexpr, GROUP_SIZE_M: gl.constexpr, dtype: gl.constexpr, num_warps: gl.constexpr, NUM_SMS: gl.constexpr):
    config = MatmulConfig(M, N, K, BLOCK_M, BLOCK_N, BLOCK_K, GROUP_SIZE_M, NUM_SMS, dtype, num_warps)

    # Create channels for data flow
    a_chnl = get_desc_channel(desc_a, num_buffers=2)
    b_chnl = get_desc_channel(desc_b, num_buffers=2)
    c_chnl = TensorMemoryChannel.alloc(config.c_shape, gl.float32, config.c_tmem_layout, num_buffers=1)
    epi_chnl = SharedMemoryChannel.alloc(config.c_shape, config.dtype, gl.constexpr(desc_c.layout), num_buffers=1)

    chnls = (a_chnl, b_chnl, c_chnl, epi_chnl)
    descs = (desc_a, desc_b, desc_c)

    # Use warp specialization to parallelize the stages
    # Create a simple default partition that just returns
    @gluon.jit
    def _default_partition(config, chnls, descs):
        # Default partition does nothing, just returns empty tuple
        return ()
    
    gl.warp_specialize((config, chnls, descs), _default_partition, (config, chnls, descs), [
        _matmul_mma,
        _matmul_load,
        _matmul_epilogue,  # 将 epilogue 作为 worker partition 使用 4 个 warps
    ], [1, 1, 4], [24, 24, 24])  # epilogue 使用 4 个 warps

    # Clean up
    a_chnl.release()
    b_chnl.release()
    c_chnl.release()
    epi_chnl.release()


# ===-----------------------------------------------------------------------===#
# Entry Point
# ===-----------------------------------------------------------------------===#


def torch_dtype_to_triton(dtype):
    if dtype == torch.float8_e5m2:
        return gl.float8e5
    return getattr(gl, str(dtype).split('.')[1])


def make_tensor_desc(x, shape, strides, block_shape, order=[1, 0]):
    layout = get_nvmma_layout(block_shape, torch_dtype_to_triton(x.dtype), order=order)
    return TensorDescriptor(x, shape=shape, strides=strides, block_shape=block_shape, layout=layout.value)


def matmul_tma(a, b, warp_specialize: bool = True):
    # Check constraints.
    assert a.shape[1] == b.shape[1], "Incompatible dimensions"  # b should be [N, K] for standard matmul
    assert a.dtype == b.dtype, "Incompatible dtypes"

    M, K = a.shape
    N, K = b.shape
    dtype = a.dtype

    c = torch.empty((M, N), device=a.device, dtype=dtype)

    # Block sizes
    BLOCK_M = 128
    BLOCK_N = 128
    BLOCK_K = 64
    GROUP_SIZE_M = 8

    # Get number of SMs
    NUM_SMS = torch.cuda.get_device_properties(a.device).multi_processor_count

    # Create tensor descriptors with appropriate layouts
    a_desc = make_tensor_desc(a, shape=[M, K], strides=[a.stride(0), a.stride(1)], block_shape=[BLOCK_M, BLOCK_K])
    # B tensor descriptor - B should be [N, K] for standard matmul, will be transposed during MMA
    b_desc = make_tensor_desc(b, shape=[N, K], strides=[b.stride(0), b.stride(1)], block_shape=[BLOCK_N, BLOCK_K])
    c_desc = make_tensor_desc(c, shape=[M, N], strides=[c.stride(0), c.stride(1)], block_shape=[BLOCK_M, BLOCK_N])

    # Grid configuration
    num_pid_m = triton.cdiv(M, BLOCK_M)
    num_pid_n = triton.cdiv(N, BLOCK_N)
    grid = min(NUM_SMS, num_pid_m * num_pid_n)

    matmul_kernel[grid](
        a_desc, b_desc, c_desc,
        M, N, K,
        BLOCK_M, BLOCK_N, BLOCK_K, GROUP_SIZE_M,
        torch_dtype_to_triton(dtype),
        num_warps=8,  # 增加到 8 个 warps 以支持 epilogue 使用 4 个 warps
        NUM_SMS=NUM_SMS
    )

    return c


# ===-----------------------------------------------------------------------===#
# Unit Tests
# ===-----------------------------------------------------------------------===#


def is_cuda():
    return triton.runtime.driver.active.get_current_target().backend == "cuda"


def is_blackwell():
    return is_cuda() and torch.cuda.get_device_capability()[0] == 10


@pytest.mark.parametrize("M", [512, 1024])
@pytest.mark.parametrize("N", [512, 1024])
@pytest.mark.parametrize("K", [512, 1024])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.skipif(not is_blackwell(), reason="Gluon matmul is only supported on Blackwell GPUs")
def test_op(M, N, K, dtype):
    device = "cuda"

    torch.manual_seed(42)
    a = torch.empty((M, K), dtype=dtype, device=device).normal_(mean=0.0, std=0.5)
    b = torch.empty((N, K), dtype=dtype, device=device).normal_(mean=0.0, std=0.5)  # B is [N, K] for standard matmul

    ref_out = torch.matmul(a, b.T)  # Use b.T for reference computation
    tri_out = matmul_tma(a, b)
    
    torch.testing.assert_close(ref_out, tri_out, atol=1e-2, rtol=0)


if __name__ == "__main__":
    # Simple test
    device = "cuda"
    M, N, K = 512, 512, 512
    dtype = torch.float16
    
    torch.manual_seed(42)
    a = torch.empty((M, K), dtype=dtype, device=device).normal_(mean=0.0, std=0.5)
    b = torch.empty((N, K), dtype=dtype, device=device).normal_(mean=0.0, std=0.5)  # B is [N, K] for standard matmul

    ref_out = torch.matmul(a, b.T)  # Use b.T for reference computation
    tri_out = matmul_tma(a, b)
    
    print(f"Reference output shape: {ref_out.shape}")
    print(f"Triton output shape: {tri_out.shape}")
    print(f"Max difference: {(ref_out - tri_out).abs().max().item()}")
    print(f"Mean difference: {(ref_out - tri_out).abs().mean().item()}")
