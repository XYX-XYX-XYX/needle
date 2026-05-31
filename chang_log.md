# FlashAttention 双 Kernel 拆分变更记录

## 背景

原来项目里只有一个 FlashAttention CUDA kernel，所有实现都集中在 `src/ndarray_backend_cuda.cu` 中，并通过 Python 侧的 `flash_attention_forward` 调用。这个 kernel 当前主要按 SM80/Ampere 路径实现。

现在需要同时保留原来的 SM80 kernel，并新增一个面向 SM90/Hopper 的 kernel。由于 SM90 版本后续会使用 WGMMA 和 TMA 这类 Hopper 专属指令，所以不能简单地把两个 kernel 都放在同一个 `.cu` 文件里再统一编译所有架构。

## 问题演进

### 1. 从单 kernel 到同时支持 SM80 / SM90

最开始的目标是：

- 保留已有 SM80 FlashAttention kernel。
- 新增 SM90 kernel 入口。
- Python 层可以选择 `auto`、`sm80`、`sm90`。
- 测试和 benchmark 框架可以分别指定 kernel。

因此新增了 kernel selector：

```python
kernel="auto" | "sm80" | "sm90"
```

其中：

- `auto`：运行时根据 GPU compute capability 自动选择。
- `sm80`：强制走 SM80 kernel。
- `sm90`：强制走 SM90 kernel。

### 2. 为什么同一个 `.cu` 文件里放两个 kernel 会有风险

如果同一个 `.cu` 文件里同时包含：

- SM80 kernel：普通 Ampere 路径。
- SM90 kernel：包含 WGMMA / TMA 指令。

并且 CMake 对这个文件同时传入：

```text
-gencode=arch=compute_80,code=sm_80
-gencode=arch=compute_90a,code=sm_90a
```

那么 NVCC 会对同一个 translation unit 分别做 SM80 和 SM90a 编译 pass。

问题是：即使运行时不会在 SM80 GPU 上调用 SM90 kernel，只要 SM90 kernel 的代码在 SM80 编译 pass 中被编译器看到并实例化，SM80 pass 就可能因为不支持 WGMMA / TMA 指令而编译失败。

所以风险不是“SM80 kernel 自己会失败”，而是“包含 SM90 专属代码的同一个 `.cu` 文件被拿去编译 SM80 目标时会失败”。

### 3. 最终方案：拆分 CUDA 文件

最终采用更安全的拆分方式：

- `src/flash_attention_sm80.cu`：只放 SM80 kernel，只用 `sm_80` 编译。
- `src/flash_attention_sm90.cu`：只放 SM90 kernel，只用 `sm_90a` 编译。
- `src/ndarray_backend_cuda.cu`：只保留 pybind、公共 dispatch、benchmark wrapper 和普通 CUDA 管理逻辑。

这样后续即使 `src/flash_attention_sm90.cu` 使用 WGMMA / TMA，也不会被 SM80 编译 pass 看到。

## 修改的文件和作用

### `CMakeLists.txt`

作用：控制 CUDA 文件如何编译，以及每个 kernel 文件对应哪个 GPU 架构。

主要变化：

- 移除了原来基于 `nvidia-smi` / `CUDA_SELECT_NVCC_ARCH_FLAGS(Auto)` 的自动架构探测。
- 新增独立架构配置：

```cmake
set(NEEDLE_CUDA_MAIN_ARCH "80" CACHE STRING "CUDA SM architecture for the pybind CUDA backend TU")
set(NEEDLE_FLASHATTN_SM80_ARCH "80" CACHE STRING "CUDA SM architecture for src/flash_attention_sm80.cu")
set(NEEDLE_FLASHATTN_SM90_ARCH "90a" CACHE STRING "CUDA SM architecture for src/flash_attention_sm90.cu")
```

- 主 pybind CUDA 文件使用：

```text
-gencode=arch=compute_80,code=sm_80
```

- SM80 FlashAttention 文件使用：

```text
-gencode=arch=compute_80,code=sm_80
```

- SM90 FlashAttention 文件使用：

```text
-gencode=arch=compute_90a,code=sm_90a
```

- 使用 `CUDA_COMPILE` 先把两个 FlashAttention `.cu` 文件分别编译成 object，再链接进最终的 `ndarray_backend_cuda` Python 扩展。
- 给手动编译出来的 FlashAttention object 增加 `-Xcompiler -fPIC`，否则链接 Python shared object 时会失败。

### `src/flash_attention_common.h`

作用：公共 CUDA 后端头文件。

包含：

- `CudaArray` 定义。
- `ELEM_SIZE`、`TILE` 等公共常量。
- `CudaCheck` 错误检查工具。
- 两个 kernel launch 声明：

```cpp
LaunchFlashAttentionForwardSm80(...)
LaunchFlashAttentionForwardSm90(...)
```

这个文件让主 pybind 文件和两个 kernel 文件共享相同的数据结构和函数声明。

### `src/flash_attention_sm80.cu`

作用：SM80 FlashAttention kernel 的独立 translation unit。

特点：

- 只用 `sm_80` 编译。
- 保留当前已有的 FlashAttention 实现路径。
- 导出：

```cpp
LaunchFlashAttentionForwardSm80(...)
```

这样原有 SM80 kernel 被保留，并且不会受到后续 SM90/WGMMA/TMA 代码影响。

### `src/flash_attention_sm90.cu`

作用：SM90 FlashAttention kernel 的独立 translation unit。

特点：

- 只用 `sm_90a` 编译。
- 当前暂时放的是与 SM80 类似的占位实现，确保接口和编译链路先打通。
- 导出：

```cpp
LaunchFlashAttentionForwardSm90(...)
```

后续真正的 Hopper WGMMA / TMA 实现应该写在这个文件里。因为它只会被 `sm_90a` 编译，所以可以安全使用 Hopper 专属指令。

### `src/ndarray_backend_cuda.cu`

作用：CUDA 后端 Python 扩展入口和公共调度逻辑。

主要变化：

- 不再包含 FlashAttention kernel 本体。
- 保留 pybind module 定义。
- 保留 `FlashAttentionForward` 和 `FlashAttentionForwardBenchmark`。
- 根据 Python 传入的 `kernel` 参数做 dispatch：

```cpp
kernel == "sm80" -> LaunchFlashAttentionForwardSm80(...)
kernel == "sm90" -> LaunchFlashAttentionForwardSm90(...)
kernel == "auto" -> 根据当前 GPU compute capability 自动选择
```

- 暴露后端元数据：

```python
__flash_attention_kernels__ = ["auto", "sm80", "sm90"]
__cuda_compute_capability__ = (major, minor)
```

### `python/needle/backend_ndarray/ndarray.py`

作用：NDArray 层 FlashAttention API。

主要变化：

- `flash_attention` 增加 `kernel` 参数：

```python
def flash_attention(q, k, v, dropout, causal, kernel="auto")
```

- 检查 `kernel` 是否属于：

```python
{"auto", "sm80", "sm90"}
```

- 调用 CUDA backend hook 时，把 `kernel` 一起传给 pybind。

### `python/needle/ops/ops_flashattention.py`

作用：autograd op 层。

主要变化：

- `FlashAttention` op 保存 `kernel` 参数。
- `compute` 时把 `kernel` 传给 NDArray 层。
- `flashattention(...)` 函数签名增加：

```python
kernel="auto"
```

### `python/needle/nn/nn_flashtransformer.py`

作用：神经网络模块层 FlashAttention wrapper。

主要变化：

- `FlashMutiHeadAttention` 初始化参数增加：

```python
kernel="auto"
```

- forward 时把 `kernel` 传到 `ops.flashattention(...)`。

这样模型代码可以显式选择：

```python
nn.FlashMutiHeadAttention(device=ndl.cuda(), kernel="sm80")
nn.FlashMutiHeadAttention(device=ndl.cuda(), kernel="sm90")
nn.FlashMutiHeadAttention(device=ndl.cuda(), kernel="auto")
```

### `tests/project/test_flashattention.py`

作用：FlashAttention 正确性测试。

主要变化：

- 测试参数增加 `kernel`：

```python
@pytest.mark.parametrize("kernel", ["auto", "sm80", "sm90"])
```

- `sm90` 测试会在非 compute capability 9.x 的机器上 skip。
- 当前测试仍然对比 PyTorch `scaled_dot_product_attention` 输出。

### `tests/project/test_flashattention_stub.py`

作用：后端 hook 和参数传递测试。

主要变化：

- fake CUDA backend 现在会检查 `kernel` 参数是否传到 backend hook。
- 增加非法 kernel 参数测试。
- 确认 `FlashMutiHeadAttention(..., kernel=...)` 能保存 selector。

### `tests/project/benchmark.py`

作用：FlashAttention benchmark 脚本。

主要变化：

- CLI 增加：

```bash
--kernel auto|sm80|sm90
```

- benchmark 调用后端时把 kernel selector 一起传入。
- 输出中打印当前 benchmark 使用的 kernel。

## 编译验证结果

已经运行：

```bash
make lib
```

结果：通过。

构建日志确认：

- `src/flash_attention_sm80.cu` 编译为独立 object。
- `src/flash_attention_sm90.cu` 编译为独立 object。
- 最终两个 object 都链接进：

```text
python/needle/backend_ndarray/ndarray_backend_cuda*.so
```

并且生成的 CMake 编译参数确认：

```text
src/flash_attention_sm80.cu -> -gencode=arch=compute_80,code=sm_80
src/flash_attention_sm90.cu -> -gencode=arch=compute_90a,code=sm_90a
```

## 后续开发建议

后续实现真正 SM90 kernel 时，只需要修改：

```text
src/flash_attention_sm90.cu
```

把当前占位实现替换为 WGMMA / TMA 路径即可。不要把 Hopper 专属指令放回 `src/ndarray_backend_cuda.cu` 或 `src/flash_attention_sm80.cu`，否则会重新引入 SM80 编译 pass 看到 SM90a 指令的问题。
