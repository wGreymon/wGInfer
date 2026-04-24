# Qwen3.5 接入记录

这个目录用于专门记录 `Qwen3.5` 在 `wGInfer` 中的接入过程，包括：

- 模型结构分析
- 配置字段整理
- 权重映射关系
- 推理路径设计
- `Transformers` 对齐过程
- C++ 后端迁移过程
- 问题定位与修复记录

后续我会把与 `Qwen3.5` 接入直接相关的内容持续写在这个目录下，而不是分散在聊天记录或零散笔记里。

---

## 当前接入目标

当前阶段围绕 `Qwen3.5-4B` 的 `text-only` 路径展开，重点目标是：

1. 理解 Hugging Face 官方实现中的模型结构
2. 明确 `linear_attention` 与 `full_attention` 的推理路径
3. 完成 `wGInfer` 中 `Qwen3.5` 的 C++ 主推理链路
4. 持续提高与 `Transformers` 的 correctness 对齐程度

---

## 建议后续文档拆分

后续可以在这个目录下继续补这些文档：

- `01_model_structure.md`
  - 记录 `Qwen3.5` 的模型结构理解

- `02_weight_mapping.md`
  - 记录 Hugging Face 权重名与 `wGInfer` 内部字段映射

- `03_inference_path.md`
  - 记录 prefill / decode / generate 的推理路径设计

- `04_alignment_log.md`
  - 记录和 `Transformers` 的对齐过程、分叉点和修复尝试

- `05_cpp_migration_log.md`
  - 记录从 Python 编排到 C++ backend 的迁移过程

- `06_host_fallback_plan.md`
  - 记录当前 C++ 后端中的 host fallback 与后续替换优先级

- `07_cpp_backend_refactor_plan.md`
  - 记录 Qwen3.5 C++ 后端重构方案与分阶段验证策略

---

## 当前认识

### 1. 当前走的是 text-only 路径

当前仓库中的 `Qwen3.5` 接入，并没有实现完整多模态分支，而是只围绕：

- `model.language_model.*`

这条语言模型主干展开。

也就是说，当前保留的是：

- 文本 embedding
- mixed transformer blocks
- final norm
- lm head / output projection

当前忽略的是：

- `model.visual.*`
- `mtp.*`
- 视觉塔与其他非文本分支

### 2. Qwen3.5 比 Qwen2 更复杂

`Qwen2` 更像标准 decoder-only transformer，结构统一、cache 形式单一。  
`Qwen3.5` 则混合了：

- `linear_attention`
- `full_attention`

并且：

- full-attention 用 `K/V cache`
- linear-attention 用 `recurrent state + qkv history`

这也是当前实现、调试和对齐难度显著更高的原因。

### 3. 当前主推理已经向 C++ backend 收口

当前 `Qwen3.5` 的方向已经不是保留 Python 主推理，而是尽量向 `Qwen2` 靠齐：

- Python 负责配置解析与权重加载
- C++ 负责主推理路径

### 4. 当前改进方向重新确认

当前阶段对 `Qwen3.5` 的改进方向重新收敛为两点：

1. **correctness 尽量对齐 Hugging Face `Transformers`**
2. **代码结构尽量清晰、职责尽量分明，便于继续维护**

这意味着当前阶段的优先级不是：

- 先追 benchmark
- 先追工业化服务引擎
- 先追更多模型接入

而是：

- 先让 C++ 后端结构更贴近 HF 的模块组织
- 再逐步提高长序列对齐能力
- 最后再做 host fallback 替换和性能优化

一句话总结：

```text
结构清晰 > correctness 对齐 > 性能优化
```

---

## 记录建议

后续每次推进 `Qwen3.5` 接入时，建议记录：

- 日期
- 目标
- 修改点
- 验证命令
- 结果
- 当前问题
- 下一步

可以使用下面的模板：

```md
## YYYY-MM-DD

### 目标

### 本次改动

- 项目 1
- 项目 2

### 验证

- 命令：
- 结果：

### 当前问题

- 项目 1

### 下一步

- 项目 1
```
