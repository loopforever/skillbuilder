# Codebase Skill Generator

Analyzes a large codebase and generates `SKILL.md` files that agentic coding assistants (Claude Code, Cursor, etc.) can use to understand your project's conventions.

**Zero external dependencies** — uses only the Python 3.10+ standard library. Talks to any OpenAI-compatible LLM server (OpenAI, Ollama, LM Studio, vLLM, llama.cpp, etc.) via raw HTTP.

## Key Features

- **Project-aware**: Each immediate subdirectory of root_dir is treated as a separate
  "project". The tool identifies which patterns are universal across all projects vs
  unique to specific projects, and documents both.
- **Naming convention extraction**: Class names, method names, field names, constants,
  parameters — casing style, prefix/suffix patterns, verb conventions.
- **Ordering convention extraction**: How fields, methods, imports, and annotations
  are ordered within a file. Skeletons include machine-readable ordering metadata.
- **Scales to thousands of files**: Hierarchical reduce, context-budget-aware batching,
  stratified sampling by project and directory, collision-free caching.

## How It Works

The pipeline has four phases, designed for LLMs with limited context windows:

```
Phase 0: Discover         Find all files matching each category's glob patterns.
                          Detect projects (immediate subdirs of root_dir).
                          Map every file to its project.
Phase 1: Extract          Build structural skeletons (no LLM) — strips method bodies,
                          keeps signatures, annotations, class structure, and adds
                          ordering + naming metadata annotations. ~5-10x compression.
Phase 1b: Sample          Stratified sampling by project first, then by directory,
                          then by complexity. Guarantees cross-project coverage.
Phase 2: Analyze          LLM analyzes each sampled file individually (one file per call).
                          Each analysis is tagged with the file's project name.
Phase 2b: Skeleton Scan   LLM scans skeletons of remaining files in batches for broad coverage.
Phase 3: Synthesize       LLM merges all analyses into a SKILL.md per category.
                          Produces universal patterns + project-specific variations.
Phase 4: Validate         (optional) LLM checks SKILL.md against unseen files, including
                          naming compliance, ordering compliance, and project accuracy.
```

Every intermediate result is cached to disk, so you can interrupt and resume with `--resume`.

### Project Structure Assumption

```
root_dir/
├── project-alpha/          # <-- project
│   └── src/main/java/...
├── project-beta/           # <-- project
│   └── src/main/java/...
└── shared-lib/             # <-- project
    └── src/main/java/...
```

Each immediate subdirectory of `root_dir` is a project. The generated SKILL.md will
document which patterns are shared across all projects and which are specific to individual
projects.

## Categories

| Key               | Glob Pattern                           | Description                          |
|-------------------|----------------------------------------|--------------------------------------|
| `java-models`     | `**/model/*.java`                      | Java model/entity classes            |
| `java-daos`       | `**/dao/*DAO.java`                     | Java Data Access Objects             |
| `java-actionbeans`| `**/action/*Action.java`               | Java ActionBean controllers          |
| `frontend`        | `**/*.css`, `**/*.js`, `**/*.mjs`, `**/*.vm` | Front-end templates & code     |

## Setup

No `pip install` needed. Just have Python 3.10+ and an LLM server.

### LLM Server

The tool talks to your LLM over HTTP using the OpenAI-compatible API by default.
It auto-detects Ollama when the URL contains port 11434 or `/ollama`.

| Server            | Example                                                                     |
|-------------------|------------------------------------------------------------------------------|
| **OpenAI**        | `python generate_skills.py /src --model gpt-4o` (default api-base)          |
| **Ollama**        | `--api-base http://localhost:11434 --model llama3.1`                        |
| **vLLM**          | `--api-base http://host:8000 --model meta-llama/Llama-3.1-8B`              |
| **SGLang**        | `--api-base http://host:30000 --model qwen2.5`                             |
| **LM Studio**     | `--api-base http://localhost:1234`                                          |
| **llama.cpp**     | `--api-base http://localhost:8080`                                          |

### Authentication

The tool reads your API key from the `OPENAI_API_KEY` environment variable automatically.
You can also pass it explicitly:

```bash
# Explicit Bearer token (sent as Authorization: Bearer <key>)
python generate_skills.py /src --api-key tok_abc123

# Or use the environment variable (auto-detected)
export OPENAI_API_KEY=sk-...
python generate_skills.py /src

# Custom auth header (e.g. X-API-Key)
python generate_skills.py /src --header "X-API-Key: my-secret-token"

# Multiple custom headers
python generate_skills.py /src --header "X-API-Key: tok" --header "X-Org-ID: myorg"

# Custom header overrides Bearer token if both set
python generate_skills.py /src --api-key ignored --header "Authorization: Token custom-scheme"
```

## Usage

### Basic (OpenAI)

```bash
export OPENAI_API_KEY=sk-...
python generate_skills.py /path/to/codebase --model gpt-4o
```

### Specifying Model and Server

```bash
# Ollama on localhost (auto-detected as Ollama by port)
python generate_skills.py /path/to/codebase \
    --api-base http://localhost:11434 \
    --model llama3.1

# vLLM on a GPU box (just pass host:port, /v1 is added automatically)
python generate_skills.py /path/to/codebase \
    --api-base http://gpu-box:8000 \
    --model meta-llama/Llama-3.1-8B-Instruct

# SGLang with auth
python generate_skills.py /path/to/codebase \
    --api-base http://localhost:30000 \
    --api-key my-token \
    --model qwen2.5

# LM Studio
python generate_skills.py /path/to/codebase \
    --api-base http://localhost:1234

# Force Ollama API protocol explicitly
python generate_skills.py /path/to/codebase \
    --api-base http://192.168.1.50:11434 \
    --api-type ollama \
    --model llama3.1
```

### Common Options

```bash
# Only process specific categories
python generate_skills.py /src --categories java-models java-daos

# Analyze more files per category for better coverage
python generate_skills.py /src --sample-size 20

# Resume an interrupted run
python generate_skills.py /src --resume

# Run validation against unseen files
python generate_skills.py /src --validate

# Faster run: skip the broad skeleton overview
python generate_skills.py /src --skip-skeleton-overview

# Start fresh (clear cached intermediate results)
python generate_skills.py /src --clean

# Custom output location
python generate_skills.py /src --output-dir ./my-skills
```

### All Options

```
positional arguments:
  root_dir               Root directory of the codebase to analyze

options:
  --model MODEL          Model name as your LLM server knows it (default: llama3.1)
  --api-base URL         Base URL of your LLM server (default: https://api.openai.com)
  --api-type TYPE        "ollama", "openai", or "auto" (default: auto — detects
                         Ollama by port 11434, otherwise uses OpenAI-compatible)
  --api-key KEY          Bearer token for Authorization header. Falls back to
                         OPENAI_API_KEY env var if not provided.
  --header 'Name: val'   Extra HTTP header(s), repeatable (default: none)
  --sample-size N        Files to deeply analyze per category (default: 20)
  --context-budget N     Your LLM's context window in tokens (default: 100000)
  --output-dir DIR       Where to write final SKILL.md files
  --work-dir DIR         Cache directory for intermediate results
  --categories [...]     Which categories to process (default: all)
  --resume               Reuse cached intermediate results
  --validate             Run validation against unseen files
  --clean                Clear cache and start fresh
  --skip-skeleton-overview  Skip broad skeleton scan (faster)
  --seed N               Random seed for sampling (default: 42)
```

## Output

```
generated-skills/
├── java-models/
│   ├── SKILL.md                # <-- Use this
│   └── VALIDATION_REPORT.md    # If --validate was used
├── java-daos/
│   └── SKILL.md
├── java-actionbeans/
│   └── SKILL.md
└── frontend/
    └── SKILL.md
```

## Intermediate Work Directory

The `.skill-gen-work/` directory stores all intermediate results:

```
.skill-gen-work/
├── projects.json                # Detected project names
├── file_to_project.json         # File -> project mapping
├── java-models/
│   ├── files.json               # All discovered file paths
│   ├── sample_selection.json    # Which files were selected (project-stratified)
│   ├── skeletons/               # Structural skeletons with ordering/naming metadata
│   ├── analyses/                # Per-file LLM analysis (tagged with project name)
│   ├── skeleton_overview.md     # Broad skeleton batch summary
│   └── SKILL_draft.md           # Synthesis result (universal + project-specific)
...
```

This means:
- You can inspect any intermediate step
- Interrupted runs resume cheaply with `--resume`
- You can manually edit an analysis and re-run synthesis
- You can tweak prompts in `prompts.py` and re-run just the synthesis phase by
  deleting the `SKILL_draft.md` files

## Customization

### Adding New Categories

Edit `CATEGORIES` in `extractors.py`:

```python
CATEGORIES = {
    ...
    "java-services": {
        "label": "Java Service",
        "description": "Service layer classes",
        "globs": ["**/service/*.java", "**/services/*Service.java"],
    },
}
```

### Tuning Prompts

All LLM prompts are in `prompts.py`. The three key functions:
- `build_analysis_prompt()` — what to look for in each file
- `build_synthesis_prompt()` — how to merge analyses into a SKILL.md
- `build_validation_prompt()` — how to check the result

### Tuning Skeleton Extraction

The extractors in `extractors.py` use regex-based heuristics. They work well for
typical Java/JS/CSS/Velocity files but may need adjustment for unusual formatting.
If a skeleton looks wrong, check the `_extract_*_skeleton()` functions.

## Tips

- **Start with `--sample-size 5` and one category** to iterate quickly on prompt tuning.
- **Check the skeletons first** — if they look wrong, the LLM analysis will be too.
  Run with `--skip-skeleton-overview` and inspect `.skill-gen-work/*/skeletons/`.
- **Validate is worth it** — the validation report often catches blind spots.
- **The SKILL.md is a starting point** — plan to manually review and refine it. The
  LLM gets you 80% of the way; your domain knowledge fills the remaining 20%.
- **Model choice matters** — larger models (34B+) produce noticeably better synthesis.
  The per-file analysis works fine with 8B models.

## Scaling to Large Codebases (1000+ files)

The tool is designed to handle thousands of files per category:

- **Stratified sampling** groups files by directory, then samples proportionally
  from each group. This ensures coverage across different parts of the codebase
  instead of clustering in one directory.
- **Hierarchical reduce** merges LLM results in a tree: batch → summarize →
  batch summaries → summarize again, until everything fits in one context window.
  Automatically adapts to your `--context-budget`.
- **Frontend filtering** aggressively excludes vendor libraries, minified bundles,
  node_modules, build output, and files over 200KB to avoid drowning in noise.
- **Collision-free caching** uses content-addressed cache keys (SHA1 of full path),
  so `com/foo/model/User.java` and `com/bar/model/User.java` never collide.
- **Resumability** — every LLM call is cached. Kill the process and `--resume`
  picks up where it left off.

### Recommended settings for large codebases

```bash
# Small context window (8k-16k tokens)
python generate_skills.py /src \
    --context-budget 8000 \
    --sample-size 25 \
    --validate

# Medium context window (32k tokens)
python generate_skills.py /src \
    --context-budget 32000 \
    --sample-size 30 \
    --validate

# Large context window (100k+ tokens) — default
python generate_skills.py /src \
    --sample-size 30 \
    --validate
```

At 8k tokens with 2000 files per category, expect roughly:
- ~2000 skeleton extractions (instant, no LLM)
- ~25 per-file LLM analyses
- ~55 skeleton overview batches → ~3 merge batches → 1 final overview
- ~3 analysis reduce batches → 1 synthesis
- ~5 validation calls
- **Total: ~90 LLM calls per category**
