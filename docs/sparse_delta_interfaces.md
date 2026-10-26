# Sparse Delta Checkpoints 接口文档

## 1. 范围与兼容等级

本文配套 [设计文档](sparse_delta_design.md)，以同一工作区基线为准，规定数据接口、v2 文件格式、现有 HTTP 协议及目标框架适配契约。

- **现有接口**：当前仓库真实的函数、类型和 endpoint，可按所列路径查阅。
- **移植接口**：为目标框架定义的建议抽象，不是当前可直接 import 的类。
- **扩展字段**：只属于建议的外层 manifest，不得误发给现有接口并假定其已校验。

兼容等级：F（v2 文件兼容）、P（stage/commit 协议兼容）、R（故障隔离与顺序重放）。生产移植至少应达到 F+P+R。首版限定有限 FP16/BF16/FP32/FP64 权重、`scale_factor=1.0`、零阈值；格式中定义了整数 dtype 不代表本规范承诺整数溢出场景的无损更新。

## 2. 现有 Python 数据接口

### 2.1 权重输入与提取

模块：[`prime_rl.utils.delta`](../src/prime_rl/utils/delta.py)。

```python
class TensorStore(Protocol):
    def keys(self) -> Iterable[str]: ...
    def get_tensor(self, name: str) -> torch.Tensor: ...

class ModelDeltaManager:
    def extract_sparse_delta_from_state_dicts(
        self, base_state: Mapping[str, torch.Tensor],
        finetuned_state: Mapping[str, torch.Tensor],
        delta_output_path: str | Path, *,
        index_encoding: str = "optimized", save_stats: bool = False,
    ) -> DeltaStats | None: ...

    def extract_sparse_delta_streaming_from_state_dicts(
        self, base_state: Mapping[str, torch.Tensor],
        finetuned_state: Mapping[str, torch.Tensor],
        delta_output_path: str | Path, *, group_size: int = 4,
        index_encoding: str = "optimized", save_stats: bool = False,
    ) -> DeltaStats | None: ...
```

| 项目 | 契约 |
|---|---|
| 输入 | 同一模型、同一传输 dtype 的 base/target；建议独立且连续的 CPU tensor，调用期间不可修改 |
| 参数集合 | 规范化后 key、shape、dtype 必须一致；当前会按特定 embedding/head 相等规则省略 tied head |
| 输出目录 | 调用者先创建父目录；路径不可指向已发布产物，当前 writer 使用覆盖写 |
| 编码 | `optimized` 或 `naive`，其他值报错 |
| `group_size` | 非负整数；0 每记录处理边界 flush，正数按 layer 编号分组 |
| 返回 | `save_stats=False` 返回 None；True 返回 `DeltaStats` |
| 副作用 | 写文件；不发布 STABLE，不推进版本，不上传，也不保存完整基座 |
| 失败 | key/shape/dtype 不匹配或无法精确重建等抛出异常；可能留下部分文件，调用者不能发布 |

`extract_sparse_delta(base_model_path, finetuned_model_path, delta_output_path, *, index_encoding="optimized", save_stats=False)` 和 `extract_sparse_delta_streaming(..., group_size=4, ...)` 是文件输入版本；base/target 参数指向单个 safetensors 文件，不是自动解析 HF 多 shard 目录的接口。

`DeltaStats` 字段：`total_params: int`、`changed_params: int`、`change_percentage: float`、`total_size_mb: float`、`delta_size_mb: float`、`compression_ratio: float`。大小以 `1024**2` 字节计；比例是 full/delta，参数统计排除规范化后跳过的参数。上述提取方法的 `save_stats=True` 表示计算并返回统计，不表示自动写 JSON；历史 `extract_delta_fused_sparse` 才有单独写统计文件的路径。

### 2.2 编解码和验证

```python
def encode_sparse_indices(indices: Tensor, *, encoding: str = "optimized") -> Tensor: ...
def decode_sparse_indices(encoded_indices: Tensor, value_count: int) -> Tensor: ...
def iter_streaming_delta_records(path: str | Path) -> Iterator[StreamingDeltaRecord]: ...
```

解码结果是展平 int64 索引。`StreamingDeltaRecord` 字段为 `name: str`、`shape: tuple[int, ...] | None`、`encoded_indices: Tensor`、`values: Tensor`、`format_version: int`；v2 必须有 shape，v1 可以是 None。record 解析器读取已完成文件，不实现等待文件增长的 tail 行为。

`verify_sparse_delta_state_dicts(base_state, target_state, delta_path, ...)` 与 `verify_sparse_delta_file(base_path, target_path, delta_path, ...)` 返回 `DeltaVerificationResult(max_diff, mismatches)`，其 `.ok` 表示没有不匹配项。精确验收显式使用 `atol=0, rtol=0, include_bias=True`；默认容差不能证明无损。

最小可执行示例（在本仓库已安装依赖的环境中使用 `uv run` 执行）：

```python
from pathlib import Path
from tempfile import TemporaryDirectory
import torch
from prime_rl.utils.delta import ModelDeltaManager, verify_sparse_delta_state_dicts

base = {"model.norm.weight": torch.tensor([1., 2., 3.], dtype=torch.bfloat16)}
target = {"model.norm.weight": torch.tensor([1., 4., 3.], dtype=torch.bfloat16)}
with TemporaryDirectory() as directory:
    path = Path(directory) / "delta.safetensors"
    stats = ModelDeltaManager().extract_sparse_delta_from_state_dicts(
        base, target, path, save_stats=True,
    )
    result = verify_sparse_delta_state_dicts(base, target, path, atol=0, rtol=0)
    assert result.ok
    assert stats is not None and stats.changed_params == 1
```

### 2.3 布局适配与应用

模块：[`weight_layout.py`](../src/prime_rl/inference/vllm/worker/weight_layout.py)。

```python
@dataclass(frozen=True)
class LogicalSparseUpdate:
    name: str
    shape: tuple[int, ...]
    indices: torch.Tensor
    values: torch.Tensor

@dataclass(frozen=True)
class LocalSparseUpdate:
    name: str
    indices: torch.Tensor
    values: torch.Tensor

class ModelLayoutAdapter(Protocol):
    def to_local(self, update: LogicalSparseUpdate) -> LocalSparseUpdate: ...
```

`to_local` 必须解析目标参数，检查布局支持程度，筛选本 rank 的更新并将索引转换为本地展平索引。返回 values 的顺序与 dtype 必须保持；不能提前 cast 成模型 dtype。没有属于本 rank 的位置时返回空 indices/values；逻辑参数无法映射时应当报错，不得静默跳过。

当前接口一个逻辑参数只返回一个本地参数。若目标引擎将一个逻辑参数拆成多个独立 Parameter，需要新增一对多 adapter 和相应 apply 循环，不能直接假定当前接口覆盖该布局。

实际应用函数：

```python
def load_sparse_delta_weights(
    model: torch.nn.Module, delta_path: str, scale_factor: float = 1.0,
) -> None: ...
```

位于 [`weight_transfer.py`](../src/prime_rl/inference/vllm/worker/weight_transfer.py)。此函数原地更新参数、不返回统计对象；内部日志记录更新数量。它不检查 base_version、不负责暂停推理、不提供回滚、也不提供重试幂等。当前实现内置 `DenseVllmModelLayoutAdapter`；接入其他引擎时应显式注入或替换 adapter，不能仅复制函数签名。

worker 包装器 `FileSystemWeightUpdateWorker.update_weights_from_delta_path(delta_path)` 支持文件或目录；目录优先选择 `delta.stream`，否则选择 `delta.safetensors`。它检查 PP/EP/量化限制，并调用 `update_mla_absorbed_weights` 刷新必要派生权重。

## 3. 目标框架适配契约

以下类型是**移植接口提案**，可由框架自己的同步或异步 API 实现。它们不改变 v2 文件格式。

```python
@dataclass(frozen=True)
class Snapshot:
    version: str
    tensors: Mapping[str, torch.Tensor]

@dataclass(frozen=True)
class DeltaArtifact:
    version: str
    base_version: str
    path: Path
    format: Literal["safetensors_v2", "stream_v2"]
    size_bytes: int
    sha256: str

class SnapshotExporter(Protocol):
    def export(self, model: object, version: str) -> Snapshot | None: ...

class DeltaPublisher(Protocol):
    def publish(self, base: Snapshot, target: Snapshot, output_dir: Path) -> DeltaArtifact: ...

class InferenceControl(Protocol):
    async def stage(self, artifact: DeltaArtifact) -> None: ...
    async def pause(self) -> None: ...
    async def commit(self, version: str) -> None: ...
    async def resume(self) -> None: ...
    async def reload_base(self) -> None: ...
```

职责和时序：

1. 所有需要的训练 rank 调用 exporter；仅产物负责人返回 Snapshot，其余返回 None。exporter 承担 gather、HF 命名、dtype、alias 和快照独立性。
2. publisher 校验版本，写入独占目录，完成文件和完整性信息后发布完成事件；失败不返回成功产物。
3. 提前 streaming 上传需要另加 `begin(version, base_version, path, done_path)` 钩子；不能等 `DeltaArtifact` 的最终 size/hash 已确定后才开始上传并宣称与提取重叠。
4. controller stage 期间允许旧版本推理；pause 后 commit。只有成功应用或确认未修改权重时才能安全 resume；dirty 时保持隔离并恢复。
5. 调度器分别管理 producer 的已发布版本和每个 consumer 的已确认版本，不能用一个全局整数代替全部节点状态。

建议增加外层 manifest，字段包括 `schema_version`、`chain_id`、`model_revision`、`base_fingerprint`、`version`、`base_version`、`optimizer_step`、`artifact_format`、`size_bytes`、`sha256` 和明确的传输 dtype/alias 规则。它们属于扩展：当前 v2 文件不记录版本、模型内容指纹或链 ID，当前 HTTP 也不验证这些扩展字段。若启用，发送端与接收端必须同时实现校验。

## 4. 文件格式契约

### 4.1 Safetensors v2

文件名：`delta.safetensors`。每个发生变化的逻辑参数 `N` 包含两个 1D tensor：

| Key | 内容 |
|---|---|
| `N.__idx` | uint8 varint 字节，或 int32/int64 原始索引 |
| `N.__val` | 与解码索引数量一致的加法差值；dtype 可以宽于权重 |

metadata 是 string 到 string 的映射：

```json
{
  "prime_rl_delta_format": "sparse_delta_v2",
  "prime_rl_delta_layout": "huggingface",
  "prime_rl_delta_shapes": "{\"model.norm.weight\": [3]}"
}
```

shape 表只需列出出现的变化参数；没有 tensor 的空 delta 仍必须保留合法 metadata。读者必须检查 idx/val 配对、shape、索引范围和顺序。参数名不得与格式保留后缀产生歧义。

### 4.2 Stream v2 二进制布局

文件名：`delta.stream`。文件头和记录头均使用小端、无对齐填充的 `struct` 编码。当前 tensor payload 通过 NumPy 原始内存写入，实际部署要求小端主机；跨大端平台移植必须显式处理字节序。

文件头 `struct.Struct("<8sI")`，共 12 字节：

| 偏移 | 类型 | 含义 |
|---|---|---|
| 0 | 8 bytes | ASCII `PDELSTRM` |
| 8 | uint32 | version，v2 为 2 |

每条记录头 `struct.Struct("<I B B H Q Q")`，共 24 字节：

| 偏移 | 类型 | 含义 |
|---|---|---|
| 0 | uint32 | UTF-8 参数名的字节数 |
| 4 | uint8 | 索引编码：0=varint，1=int32，2=int64 |
| 5 | uint8 | values dtype code |
| 6 | uint16 | tensor rank，即 shape 维数 |
| 8 | uint64 | 编码后的索引 payload 字节数 |
| 16 | uint64 | values 元素数量，不是字节数 |

记录体依次是：参数名字节、`rank` 个 uint64 shape 维度、索引 payload、values payload。values 字节数等于 `value_count × dtype_size`，无 padding、无结束哨兵。合法标量 rank=0、shape=()。

dtype code：1=float16，2=bfloat16，3=float32，4=float64，5=int64，6=int32，7=int8，8=uint8。BF16 保存原始 16 位表示。写入器只为变化 tensor 写 record；完全不变的模型可只写 12 字节文件头。同一逻辑参数不得重复出现。

v1 使用相同记录头但偏移 6 是必须为零的保留字段，记录体没有 shape。新接收端可保留 v1 支持，新发送端只写 v2。

### 4.3 Varint 索引

对递增索引 `x[0..k-1]`，令 `previous=0`，依次编码 `x[j]-previous`，再令 `previous=x[j]`。每字节低 7 位携带 payload，高位为后续字节标记，低位组先写。解码后做累加和。

例如 `[100, 103, 104, 110]` 的间隔为 `[100, 3, 1, 6]`，字节为十六进制 `64 03 01 06`。索引 0 合法，首个间隔可以为零；后续间隔必须为正。当前 int64 解码约束单个间隔不超过 9 字节。

naive 编码在最大索引 `< 2**31` 时用 int32，否则用 int64。原始索引 payload 长度必须分别能被 4 或 8 整除。`decode_sparse_indices` 的 `value_count`（底层 varint 解码器称为 `expected_count`）来自 values 元素数，不是压缩字节数。

## 5. HTTP 控制协议

实现：[`server.py`](../src/prime_rl/inference/vllm/server.py)；参考调用方：[`utils/client.py`](../src/prime_rl/utils/client.py)。以下为管理服务路由，不是 OpenAI `/v1` 推理接口。调用方负责现有部署的管理地址与认证；不要假定公开推理地址自动授权管理操作。

### 5.1 公共字段与返回

| 字段 | 类型 | 要求 |
|---|---|---|
| `version` | string | 调用方必须提供；当前验证器部分入口缺省为 `unknown`，不得依赖该缺省 |
| `mode` | `full` / `delta` | 移植调用必须显式传 `delta`，现有缺省为 full |
| `base_version` | string | delta 必填，首次为 `"0"`，必须等于当前 active version |
| `relay` | boolean 或可解析的布尔字段 | 可选，是否转发给已配置 peer；内部转发使用 false 防循环 |

成功返回 HTTP 200 与 `status="ok"`；pause/resume 分别为 `paused`/`resumed`。额外 relay/timing 字段可忽略，但不能忽略非 2xx。常见校验错误为 `{"error": "说明"}`；worker 未捕获异常的 5xx 响应不保证同样 JSON 结构。

### 5.2 非 streaming stage

`POST /stage` 支持 JSON 或 form 字段；文件上传使用 multipart 的 `file` 字段。

共享路径示例：

```json
{"version":"1","base_version":"0","mode":"delta","path":"/shared/run/v1/delta.safetensors"}
```

路径必须对服务端和实际 worker 可见。上传示例字段为 `version=1`、`base_version=0`、`mode=delta` 加 `file=@delta.safetensors`。服务端保存后返回 `status`、`version`、`mode`、`path`、`ingest_ms`、`relay`、`fanout_ms`。

此接口登记 staging，不解析并验证所有 tensor，也不提供整个文件 SHA-256 参数校验。共享路径产物必须保持不变直到应用和所需重试结束。

### 5.3 Streaming upload

| 步骤 | Endpoint | 请求 | 关键响应 |
|---|---|---|---|
| 初始化 | `POST /stage_stream_init` | JSON/form：公共字段 + `filename` | `upload_id`, `version`, `mode` |
| 写块 | `POST /stage_stream_chunk` | query：`upload_id`, `offset`；body：原始非空 bytes | `received_bytes` |
| 完成 | `POST /stage_stream_finalize` | JSON/form：`upload_id`, `final_size`, `sha256` | `version`, `mode`, `path` |

init 示例：

```json
{"version":"1","base_version":"0","mode":"delta","filename":"delta.stream"}
```

chunk 示例：`POST /stage_stream_chunk?upload_id=<id>&offset=0`，body 是文件从该 offset 开始的字节；offset 是非负字节位置。参考客户端还传 `chunk_index`，当前接收端不靠它决定顺序。服务端按 offset 写文件并合并已接收区间，允许重传相同区间；发送端必须保证相同区间内容一致并避免并发冲突写入。

finalize 示例：

```json
{"upload_id":"<id>","final_size":4096,"sha256":"<64位十六进制摘要>"}
```

`final_size` 必须非负且等于最终文件大小。服务端要求范围恰好覆盖 `[0, final_size)`，并验证实际大小；sha256 在现有服务端可选，但本移植契约要求发送，参考客户端也始终发送。成功后消费 upload 状态并登记 staged version；重复 finalize 不保证幂等，可能返回 404。

如果文件仍在生成，调用方必须等待生产者完成事件。临时 EOF 不能视为完成。当前 Python uploader 用 `done_path=.../STABLE` 判断完成；不给 done_path 就是上传已完成文件的语义。

### 5.4 已知长度的 chunked upload

这是独立于 streaming 的路径，用于已完成文件。

- `POST /stage_chunk`：multipart，`file` 是块；form 包含公共字段、`filename`、`offset`、`total_size`、`sha256`。
- `POST /stage_finalize`：JSON/form，包含公共字段、`filename`、`total_size`、`sha256`。

没有 upload_id；当前上传以 `(version, mode)` 为 key。同一上传的 filename、base_version、大小及 hash 必须一致。块不能超过 total_size；finalize 检查接收区间覆盖和 SHA-256。成功登记 staged version 后才能 commit。

### 5.5 Commit、暂停与恢复

`POST /commit`，JSON/form：

```json
{"version":"1","mode":"delta"}
```

成功响应包含 `status="ok"`、`active_version="1"`、`mode="delta"`，以及 relay/timing。服务端根据已 staged 的 entry 获取 base_version 和路径，持锁再次校验，设置 dirty 后调用全部 worker，成功才推进 active 并清除 dirty。

调用序列必须是：

```text
stage 完成 → POST /pause → POST /commit → 成功后 POST /resume
```

`/pause` 当前固定调用引擎 `pause_generation(mode="keep", clear_cache=False)`；客户端 query 不改变该实现。`/resume` 在 dirty 时返回 409。当前 Python 包装器在 finally 中尝试 resume，但 dirty 时服务端会拒绝；移植时必须保留此拒绝能力，不能无条件恢复故障模型。

`POST /reload_weights` 可无 body，或携带 relay 字段。它重载引擎配置的 base，清理 staged/upload 状态，active_version 变为 None（按 0 解释）。先暂停并隔离节点，再 reload、逐版本重放、确认目标版本后重新路由。此接口不是从任意 checkpoint 路径恢复的通用 load API。

`POST /update_weights` 是 legacy direct 路径，JSON 为 `weight_dir` 和 `mode`；它不具备 stage_commit 的版本和 dirty 保障，不用于本移植规范的生产主路径。

### 5.6 错误与重试规则

| 返回/情况 | 示例 | 行为 |
|---|---|---|
| 400 | 缺字段、非法 mode/offset、多个 API server、chunk 越界 | 修正请求，禁止盲重试 |
| 404 | 不存在的 upload、path 或 staged version | 确认生命周期；必要时重新 stage 或恢复基座 |
| 409 | base 不匹配、dirty、未完整上传、hash 错误、元数据冲突 | 根据原因恢复；不得仅改版本号 |
| 5xx/连接中断 | worker 失败、relay 失败或响应丢失 | 结果可能不确定，隔离节点；在状态仍可信时重试 commit，否则 reload/replay |

重复提交**当前 active 且仍有 staging entry** 的版本不重复本地应用。跨服务进程重启、旧版本条目删除、相同 version 不同内容不在现有持久幂等保证范围内。不能通过重新 stage 一个已激活 delta 来实现通用重试。

## 6. 配置契约

共享 RL 配置示例：

```toml
[weight_broadcast]
type = "filesystem"
mode = "delta"
update_protocol = "stage_commit"
stage_transport = "streaming_upload"
background_stage = true
retain_all_deltas = true
delta_stream_group_size = 4
```

| 字段 | 当前默认值 | 所有者与限制 |
|---|---|---|
| `mode` | `full` | trainer/orchestrator 必须一致 |
| `delta_index_encoding` | `optimized` | trainer 专用配置，`optimized`/`naive`；共享配置构造 trainer 时不显式传该字段 |
| `delta_streaming_enabled` | false | trainer 专用；共享配置从 streaming_upload 自动推导，单独启动 trainer 时显式设置 |
| `delta_stream_group_size` | 4 | trainer，>=0 |
| `retain_all_deltas` | false | trainer，只能在 delta 模式开启；lease recovery 要求 true |
| `update_protocol` | `direct` | orchestrator，移植主路径显式设 stage_commit |
| `stage_transport` | `shared_fs` | shared_fs/http_upload/chunked_upload/streaming_upload |
| `background_stage` | false | orchestrator，要求 stage_commit |

当前 scheduler 的非 shared_fs stage 传输只支持 delta。推理端要求 PP=1、无 EP、无量化；stage_commit 每 endpoint 仅一个 API server。初始 full 基座需单独配置并对齐。改变广播 dtype 必须显式评审，不能改动训练 optimization/reduce dtype 来追求稀疏率。

## 7. 接口验收与运行示例

现有针对性测试入口：

```bash
uv run pytest tests/unit/utils/test_delta.py tests/unit/inference/test_weight_layout.py tests/unit/inference/test_weight_transfer.py
uv run pytest tests/unit/train/rl/test_filesystem_broadcast.py tests/unit/inference/test_stage_protocol.py tests/unit/utils/test_client.py
uv run scripts/verify_sparse_delta.py --base /data/base.safetensors --target /data/target.safetensors --delta /data/delta.safetensors --atol 0 --rtol 0
```

依赖按仓库的 uv 环境安装；以上是验收命令，不代表已在本次文档任务执行。核心编码模块依赖 PyTorch、NumPy、Numba 和 safetensors；vLLM 仅属于当前推理适配层，目标框架不必因此引入整个训练运行时。

验收记录至少包括：支持的 dtype/布局、模型 revision、快照版本、base/target shape 清单、artifact hash、各 rank 与完整加载结果的对比、失败后的隔离与恢复结果。离线全局权重一致不等于本地 TP 投影正确；必须分别验收。参数一致也不意味着随机采样生成文本逐字一致，应使用适当的确定性 logits 检查和独立 rollout 缓存测试。
