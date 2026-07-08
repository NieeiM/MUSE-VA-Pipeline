# MUSE-VA Pipeline

[![arXiv](https://img.shields.io/badge/arXiv-Paper-b31b1b?logo=arxiv)]()
[![huggingface dataset](https://img.shields.io/badge/HuggingFace-Dataset-orange?logo=huggingface)](https://huggingface.co/datasets/jiahaomei/MUSE-VA)
[![Web Demo](https://img.shields.io/badge/Website-Demo-181717?logo=google-chrome)](https://nieeim.github.io/MUSE-VA/)
[![GitHub](https://img.shields.io/badge/GitHub-NieeiM%2FMUSE--VA--Pipeline-blue?logo=github)](https://github.com/NieeiM/MUSE-VA-Pipeline)

[**English**](./README.md) | [**中文**](README.zh-CN.md)

This repository contains the five-stage text distillation pipeline used for
constructing the MUSE-VA dataset. It includes the code, prompts, and knowledge
bases required to generate structured text annotations from Valence-Arousal
(VA) coordinates.

This repository only outputs music captions, visual prompts, and structured
annotations. Music and image generation are provided as placeholder interfaces
so that third-party audio or image generation services can be connected later.

## What's Included

- Stage 1: rule-based VA-to-emotion/theme mapping using Cowen and ANEW resources
- Stage 2: LLM-generated music attributes constrained by AudioSet/FMA knowledge bases
- Stage 3: LLM-generated full music captions and tag captions
- Stage 4: LLM-generated visual imagery, visual tags, and image-generation captions
- Stage 5: LLM-based cross-modal consistency checking
- English and Chinese prompt templates
- Configurable LLM interfaces: `mock`, `gemini`, `openai-compatible`
- Empty music and image generation interfaces

## Installation

```bash
pip install -r requirements.txt
```

## LLM Configuration

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

This interface is designed for services compatible with the OpenAI chat
completions format.

```bash
export OPENAI_COMPATIBLE_API_KEY="your_api_key"
export OPENAI_COMPATIBLE_BASE_URL="https://api.example.com/v1"
export OPENAI_COMPATIBLE_MODEL="your-model-name"

python -m muse_va_pipeline \
  --llm-provider openai-compatible \
  --num-samples 10 \
  --language en
```

The same values can also be passed through command-line arguments:

```bash
--llm-api-key
--llm-base-url
--llm-model
```

## Common CLI Options

```text
--num-samples, -n       Number of random VA samples
--valence              Fixed valence value in [1, 9]
--arousal              Fixed arousal value in [1, 9]
--language             Prompt language: en or zh
--llm-provider         mock, gemini, or openai-compatible
--llm-model            Model name
--llm-api-key          API key
--llm-base-url         Base URL for OpenAI-compatible APIs
--output-dir           Output directory
--max-workers, -w      Number of parallel workers
--stage-retries        Number of retries for LLM stages
--sleep-time           Sleep seconds between stages
--seed                 Random seed
```

## Output Fields

The final CSV contains:

- `valence`, `arousal`, `innovation`
- `emotion`, `theme`
- `genre`, `lead_instruments`, `supporting_instruments`, `tempo`, `key`
- `composition_notes`
- `caption_full`, `caption_tags`, `vocal`
- `visual_imagery`, `visual_tags`, `visual_caption`
- `consistency_result`, `consistency_reason`
- `prompt_language`, `llm_provider`, `llm_model`

Only samples that pass Stage 5 consistency checking are aggregated into
`global_summary.csv`.

## Music and Image Generation

This repository does not include concrete music or image generation backends.
Placeholder interfaces are located at:

`muse_va_pipeline/generators/base.py`

- `MusicGenerator`
- `ImageGenerator`

To connect Suno, Qwen-Image, or other generation systems, implement these
interfaces in downstream code.

## License

Apache License 2.0.
