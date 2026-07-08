# MUSE-VA Pipeline

这是 MUSE-VA 数据集构造所用五阶段文本蒸馏流程代码。仓库包含从Valence-Arousal (VA) 坐标生成结构化文本标注所需的代码、prompt 和知识库。

本仓库只输出音乐 caption、视觉 prompt 和结构化标注，并提供音乐生成和图片生成占位接口，方便后续接入第三方音频或图像生成服务。

## 包含内容

- Stage 1：基于 Cowen 和 ANEW 资源的 VA 到情绪/主题词规则映射
- Stage 2：LLM 生成音乐属性，并使用 AudioSet/FMA 知识库约束
- Stage 3：LLM 生成音乐完整 caption 和 tag caption
- Stage 4：LLM 生成视觉意象、视觉标签和图片生成 caption
- Stage 5：LLM 进行跨模态一致性检查
- 中英文 prompt 模板
- 可配置 LLM 接口：`mock`、`gemini`、`openai-compatible`
- 音乐和图片生成的空接口

## 安装

`pip install -r requirements.txt`

## LLM 配置

### Gemini

```bash
export GEMINI_API_KEY="your_api_key"
export GEMINI_MODEL="models/gemini-2.5-pro"

python -m muse_va_pipeline \
  --llm-provider gemini \
  --num-samples 10 \
  --language en
```

### OpenAI-Compatible API

该接口适用于兼容 OpenAI chat completions 格式的服务。

```bash
export OPENAI_COMPATIBLE_API_KEY="your_api_key"
export OPENAI_COMPATIBLE_BASE_URL="https://api.example.com/v1"
export OPENAI_COMPATIBLE_MODEL="your-model-name"

python -m muse_va_pipeline \
  --llm-provider openai-compatible \
  --num-samples 10 \
  --language en
```

也可以用命令行参数传入：

```bash
--llm-api-key
--llm-base-url
--llm-model
```

## 常用命令行参数

```text
--num-samples, -n       随机 VA 样本数量
--valence              固定 valence，范围 [1, 9]
--arousal              固定 arousal，范围 [1, 9]
--language             prompt 语言：en 或 zh
--llm-provider         mock、gemini 或 openai-compatible
--llm-model            模型名称
--llm-api-key          API key
--llm-base-url         OpenAI-compatible API 的 base URL
--output-dir           输出目录
--max-workers, -w      并行 worker 数
--stage-retries        LLM 阶段重试次数
--sleep-time           阶段间隔秒数
--seed                 随机种子
```

## 输出字段

最终 CSV 包含：

- `valence`, `arousal`, `innovation`
- `emotion`, `theme`
- `genre`, `lead_instruments`, `supporting_instruments`, `tempo`, `key`
- `composition_notes`
- `caption_full`, `caption_tags`, `vocal`
- `visual_imagery`, `visual_tags`, `visual_caption`
- `consistency_result`, `consistency_reason`
- `prompt_language`, `llm_provider`, `llm_model`

只有通过 Stage 5 一致性检查的样本会被聚合到 `global_summary.csv`。

## 音乐和图片生成

本仓库不包含具体音乐或图片生成后端。占位接口位于
`muse_va_pipeline/generators/base.py`：

- `MusicGenerator`
- `ImageGenerator`

如果需要接入 Suno、Qwen-Image或其他生成系统，可以在下游代码中实现这些接口。

## 许可证

Apache License 2.0。
