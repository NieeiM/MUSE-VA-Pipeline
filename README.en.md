# MUSE-VA Pipeline

Clean implementation of the five-stage text distillation pipeline used to
construct MUSE-VA. The release includes code, prompts, and knowledge bases for
producing structured text annotations from Valence-Arousal (VA) coordinates.

Music and image generation backends are intentionally left unimplemented. This
repository outputs captions/prompts and provides placeholder interfaces for users
to connect their own audio or image generation services.

## What Is Included

- Stage 1: rule-based VA-to-emotion/theme mapping with Cowen and ANEW resources
- Stage 2: LLM-based music attribute association with AudioSet/FMA constraints
- Stage 3: LLM-based music caption and tag generation
- Stage 4: LLM-based visual imagery and image prompt generation
- Stage 5: LLM-based cross-modal consistency checking
- English and Chinese prompt templates
- Configurable LLM adapters: `mock`, `gemini`, and `openai-compatible`
- Placeholder interfaces for music and image generation

## Repository Layout

```text
MUSE-VA-Pipeline/
├── muse_va_pipeline/
│   ├── knowledge_base/       # Cowen, ANEW, AudioSet, FMA resources
│   ├── prompts/              # Stage 2-5 prompt templates in English/Chinese
│   ├── pipeline/             # Five-stage distillation modules and LLM client
│   ├── utils/                # VA mapping utilities
│   ├── generators/           # Empty generation backend interfaces
│   └── main.py               # CLI entry point
├── README.en.md
├── README.zh-CN.md
├── pyproject.toml
└── requirements.txt
```

## Installation

```bash
cd MUSE-VA-Pipeline
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

For editable installation:

```bash
pip install -e .
```

## Quick Start

Run a local smoke test with the built-in mock LLM:

```bash
python -m muse_va_pipeline --llm-provider mock --num-samples 1 --output-dir logs
```

Run one fixed VA coordinate:

```bash
python -m muse_va_pipeline \
  --valence 7.5 \
  --arousal 6.0 \
  --language en \
  --llm-provider mock \
  --output-dir logs
```

Successful entries are written to:

```text
logs/<run_id>/summary.csv
logs/global_summary.csv
```

Each run also stores per-stage JSON outputs and metadata under `logs/<run_id>/`.

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

Use this adapter for OpenAI-compatible chat completion services.

```bash
export OPENAI_COMPATIBLE_API_KEY="your_api_key"
export OPENAI_COMPATIBLE_BASE_URL="https://api.example.com/v1"
export OPENAI_COMPATIBLE_MODEL="your-model-name"

python -m muse_va_pipeline \
  --llm-provider openai-compatible \
  --num-samples 10 \
  --language en
```

The same values can also be passed through:

```bash
--llm-api-key
--llm-base-url
--llm-model
```

## Main CLI Options

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
--max-workers, -w      Parallel workers
--stage-retries        Retry count for LLM stages
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

Only entries that pass Stage 5 consistency checking are aggregated into
`global_summary.csv`.

## Music and Image Generation

This release does not include any concrete music or image generation backend.
See `muse_va_pipeline/generators/base.py` for placeholder interfaces:

- `MusicGenerator`
- `ImageGenerator`

Implement these interfaces in your own downstream code if you want to connect
Suno, Qwen-Image, Stable Diffusion, or other generation systems.

## License

Apache License 2.0.

