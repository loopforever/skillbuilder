#!/usr/bin/env python3
"""
Codebase Skill Generator
=========================
Analyzes a large codebase and produces SKILL.md files for agentic coding clients.

Uses a three-phase approach:
  1. Discover & extract structural skeletons (no LLM needed)
  2. Per-file LLM analysis on sampled files
  3. Synthesis of all analyses into a SKILL.md

Intermediate results are saved to disk so you can resume interrupted runs.

Zero external dependencies — uses only the Python standard library.
Talks to any OpenAI-compatible API (Ollama, LM Studio, vLLM, llama.cpp, etc.)
via raw HTTP.

Usage:
    python generate_skills.py /path/to/codebase
    python generate_skills.py /path/to/codebase --model llama3.1
    python generate_skills.py /path/to/codebase --api-base http://localhost:11434
    python generate_skills.py /path/to/codebase --sample-size 15 --output-dir ./my-skills
    python generate_skills.py /path/to/codebase --categories java-models java-daos
    python generate_skills.py /path/to/codebase --resume  # pick up where you left off
    python generate_skills.py /path/to/codebase --parallel 8  # 8 concurrent LLM requests
"""

import argparse
import hashlib
import json
import os
import random
import shutil
import sys
import threading
import time
import urllib.error
import urllib.request
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from extractors import (
    discover_files, discover_projects, get_file_project,
    extract_skeleton, CATEGORIES,
)
from prompts import (
    build_analysis_prompt,
    build_synthesis_prompt,
    build_validation_prompt,
)


# ---------------------------------------------------------------------------
# LLM helpers — stdlib only, no external packages
# ---------------------------------------------------------------------------

# Module-level config set from CLI args in main()
_llm_config = {
    "api_base": "https://local.host",
    "api_type": "openai",
    "model": "llama3.1",
    "api_key": "",
    "extra_headers": {},                   # Additional HTTP headers
}


def configure_llm(
    api_base: str,
    model: str,
    api_type: str = "auto",
    api_key: str = "",
    extra_headers: Optional[Dict] = None,
):
    """
    Set the module-level LLM configuration.

    api_type: "ollama", "openai", or "auto" (detect from api_base).
    api_base: Server URL. Trailing /v1 is handled automatically — pass the
              bare host:port (e.g. http://localhost:8000) and the code appends
              the right path. If you include /v1 it won't be doubled.
    extra_headers: Additional HTTP headers sent with every request,
                   e.g. {"X-API-Key": "tok_abc"}.
    """
    _llm_config["api_base"] = api_base.rstrip("/")
    _llm_config["model"] = model
    _llm_config["api_key"] = api_key or os.environ.get("OPENAI_API_KEY", "")
    _llm_config["extra_headers"] = extra_headers or {}

    if api_type != "auto":
        _llm_config["api_type"] = api_type
    else:
        # Auto-detect: if the URL looks like Ollama, use Ollama's endpoint.
        # Otherwise assume OpenAI-compatible (vLLM, SGLang, LM Studio, etc.)
        base = _llm_config["api_base"].lower()
        if base.endswith(":11434") or "/ollama" in base:
            _llm_config["api_type"] = "ollama"
        else:
            _llm_config["api_type"] = "openai"

    # Print resolved config for visibility
    resolved_url = _resolve_endpoint_url()
    print(f"  LLM endpoint: {resolved_url}")
    print(f"  API type:     {_llm_config['api_type']}")
    print(f"  Model:        {_llm_config['model']}")
    if _llm_config["api_key"]:
        print(f"  Auth:         Bearer token set")
    if _llm_config["extra_headers"]:
        print(f"  Headers:      {', '.join(_llm_config['extra_headers'].keys())}")


def _resolve_endpoint_url() -> str:
    """Build the full endpoint URL, avoiding double /v1 paths."""
    base = _llm_config["api_base"]

    if _llm_config["api_type"] == "ollama":
        return f"{base}/api/chat"

    # OpenAI-compatible: base might already end with /v1
    if base.endswith("/v1"):
        return f"{base}/chat/completions"
    else:
        return f"{base}/v1/chat/completions"


def _build_headers() -> dict:
    """Build HTTP headers including auth and any custom headers."""
    headers = {"Content-Type": "application/json"}

    if _llm_config["api_key"]:
        headers["Authorization"] = f"Bearer {_llm_config['api_key']}"

    # Extra headers override defaults if there's a collision (intentional —
    # lets you replace Authorization with a custom scheme if needed)
    headers.update(_llm_config["extra_headers"])

    return headers


def _build_request_ollama(messages: List[dict], temperature: float) -> tuple:
    """Build an Ollama /api/chat request. Returns (url, payload_bytes, headers)."""
    url = _resolve_endpoint_url()
    payload = {
        "model": _llm_config["model"],
        "messages": messages,
        "stream": False,
        "options": {
            "temperature": temperature,
            "num_predict": 4096,
        },
    }
    data = json.dumps(payload).encode("utf-8")
    return url, data, _build_headers()


def _build_request_openai(messages: List[dict], temperature: float) -> tuple:
    """Build an OpenAI-compatible /v1/chat/completions request."""
    url = _resolve_endpoint_url()
    payload = {
        "model": _llm_config["model"],
        "messages": messages,
        "temperature": temperature,
        "max_tokens": 4096,
        "stream": False,
    }
    data = json.dumps(payload).encode("utf-8")
    return url, data, _build_headers()


def _parse_response_ollama(body: dict) -> str:
    """Extract text from an Ollama response."""
    return (body.get("message", {}).get("content") or "").strip()


def _parse_response_openai(body: dict) -> str:
    """Extract text from an OpenAI-compatible response."""
    choices = body.get("choices", [])
    if not choices:
        raise ValueError(f"Empty choices in response: {json.dumps(body)[:500]}")
    return (choices[0].get("message", {}).get("content") or "").strip()


def call_llm(prompt: str, system: str = "", max_retries: int = 3) -> str:
    """
    Call the configured LLM endpoint. Retries on transient failures.
    Uses only urllib from the standard library.
    """
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    api_type = _llm_config["api_type"]
    if api_type == "ollama":
        build_req = _build_request_ollama
        parse_resp = _parse_response_ollama
    else:
        build_req = _build_request_openai
        parse_resp = _parse_response_openai

    url, data, headers = build_req(messages, temperature=0.3)

    for attempt in range(1, max_retries + 1):
        try:
            req = urllib.request.Request(url, data=data, headers=headers, method="POST")
            # Long timeout — local models can be slow
            with urllib.request.urlopen(req, timeout=600) as resp:
                resp_body = resp.read().decode("utf-8")
            body = json.loads(resp_body)
            result = parse_resp(body)
            if not result:
                raise ValueError("Empty response from LLM")
            return result
        except urllib.error.HTTPError as e:
            # Read the error body for better diagnostics
            err_body = ""
            try:
                err_body = e.read().decode("utf-8", errors="replace")[:500]
            except Exception:
                pass
            msg = f"HTTP {e.code}: {e.reason}"
            if err_body:
                msg += f"\n  Response: {err_body}"
            if attempt == max_retries:
                raise RuntimeError(
                    f"LLM call failed after {max_retries} attempts: {msg}\n"
                    f"  url: {url}\n"
                    f"  Is your LLM server running at {_llm_config['api_base']}?"
                ) from e
            wait = 2 ** attempt
            print(f"  [retry {attempt}/{max_retries}] {msg} — waiting {wait}s")
            time.sleep(wait)
        except (urllib.error.URLError, OSError) as e:
            if attempt == max_retries:
                raise RuntimeError(
                    f"LLM call failed after {max_retries} attempts: {e}\n"
                    f"  url: {url}\n"
                    f"  Is your LLM server running at {_llm_config['api_base']}?"
                ) from e
            wait = 2 ** attempt
            print(f"  [retry {attempt}/{max_retries}] {e} — waiting {wait}s")
            time.sleep(wait)
        except Exception as e:
            if attempt == max_retries:
                raise
            wait = 2 ** attempt
            print(f"  [retry {attempt}/{max_retries}] {e} — waiting {wait}s")
            time.sleep(wait)


# ---------------------------------------------------------------------------
# Pipeline phases
# ---------------------------------------------------------------------------

# Rough chars-per-token estimate. Conservative (real is ~3.5 for code).
CHARS_PER_TOKEN = 3.5

# Module-level token budget, set from --context-budget CLI arg
_token_budget = 100_000

# Module-level max skill lines, set from --max-skill-lines CLI arg
_max_skill_lines = 800

# Module-level max parallel LLM requests, set from --parallel CLI arg
_max_workers = 4

# Lock for thread-safe console output
_print_lock = threading.Lock()


def _truncate_source(text: str, max_chars: int) -> str:
    """Truncate source keeping the beginning and a tail portion."""
    if len(text) <= max_chars:
        return text
    half = max_chars // 2
    return text[:half] + "\n\n... [truncated] ...\n\n" + text[-(half // 2):]


def _char_budget() -> int:
    """Max chars to put in a single prompt, leaving room for system + output."""
    # Reserve 30% for the system prompt template + model output
    return int(_token_budget * CHARS_PER_TOKEN * 0.70)


def _cache_key(filepath: str) -> str:
    """
    Deterministic, collision-free cache key from a file path.
    Uses hashlib to avoid filename collisions (e.g. two User.java in
    different packages) and keeps it short enough for any filesystem.
    """
    name = Path(filepath).stem
    h = hashlib.sha1(filepath.encode()).hexdigest()[:12]
    return f"{name[:60]}_{h}"


def _elapsed(start: float) -> str:
    """Format elapsed time since `start`."""
    secs = time.time() - start
    if secs < 60:
        return f"{secs:.0f}s"
    return f"{secs/60:.1f}m"


def phase_discover(
    root_dir: str,
    category_keys: List[str],
    project_whitelist: Optional[List[str]] = None,
) -> Tuple[dict, List[str], dict]:
    """
    Phase 0: find all files per category, detect projects, and map files to projects.
    If project_whitelist is provided, only files belonging to those projects are kept.
    Returns (files_by_cat, project_names, file_to_project).
    """
    all_files = discover_files(root_dir)
    files_by_cat = {k: v for k, v in all_files.items() if k in category_keys}

    projects = discover_projects(root_dir)

    # Build file -> project mapping
    file_to_project = {}
    for cat, paths in files_by_cat.items():
        for p in paths:
            file_to_project[p] = get_file_project(p, root_dir)

    # Apply project whitelist filter
    if project_whitelist is not None:
        whitelist_set = set(project_whitelist)
        unknown = whitelist_set - set(projects)
        if unknown:
            print(
                f"  Warning: unknown project(s): {', '.join(sorted(unknown))}. "
                f"Available: {', '.join(projects)}",
                file=sys.stderr,
            )
        projects = [p for p in projects if p in whitelist_set]
        for cat in files_by_cat:
            files_by_cat[cat] = [
                p for p in files_by_cat[cat]
                if file_to_project.get(p) in whitelist_set
            ]
        file_to_project = {
            p: proj for p, proj in file_to_project.items()
            if proj in whitelist_set
        }

    return files_by_cat, projects, file_to_project


def phase_extract_skeletons(
    files_by_cat: dict,
    work_dir: Path,
) -> dict:
    """Phase 1: extract structural skeletons (no LLM). Returns skeleton text per file."""
    skeletons = {}
    for cat, paths in files_by_cat.items():
        cat_dir = work_dir / cat / "skeletons"
        cat_dir.mkdir(parents=True, exist_ok=True)
        skeletons[cat] = {}

        t0 = time.time()
        for i, p in enumerate(paths):
            cache_file = cat_dir / (_cache_key(p) + ".txt")
            if cache_file.exists():
                skeletons[cat][p] = cache_file.read_text()
                continue

            skel = extract_skeleton(p, cat)
            cache_file.write_text(skel)
            skeletons[cat][p] = skel

            # Progress every 200 files
            if (i + 1) % 200 == 0:
                print(f"    {cat}: {i+1}/{len(paths)} skeletons ({_elapsed(t0)})")

    return skeletons


def phase_sample(
    files_by_cat: dict,
    skeletons: dict,
    sample_size: int,
    work_dir: Path,
    file_to_project: dict = None,
    recent: bool = False,
) -> dict:
    """
    Pick representative files per category.

    When *recent* is True the N most recently modified files are chosen
    (by filesystem mtime).  Otherwise uses stratified random sampling that
    guarantees coverage across different projects and codepath areas.
    """
    sampled = {}
    for cat, paths in files_by_cat.items():
        selection_file = work_dir / cat / "sample_selection.json"
        if selection_file.exists():
            selected = json.loads(selection_file.read_text())
            selected = [s for s in selected if s in skeletons.get(cat, {})]
        elif recent:
            selected = _recent_sample(paths, sample_size)
        else:
            selected = _stratified_sample(
                paths, skeletons.get(cat, {}), sample_size,
                file_to_project=file_to_project,
            )
            selection_file.parent.mkdir(parents=True, exist_ok=True)
            selection_file.write_text(json.dumps(selected, indent=2))

        # Report coverage
        n_dirs = len(set(str(Path(p).parent) for p in selected))
        n_projects = len(set(
            file_to_project.get(p, "?") for p in selected
        )) if file_to_project else 0
        proj_info = f" across {n_projects} projects," if n_projects > 1 else ""
        sampled[cat] = selected
        print(f"  {cat}: sampled {len(selected)} of {len(paths)} files{proj_info} {n_dirs} directories")

    return sampled


def _stratified_sample(
    paths: List[str],
    skeletons: dict,
    n: int,
    file_to_project: dict = None,
) -> List[str]:
    """
    Two-level stratified sampling:
      1. Allocate slots across projects proportionally
      2. Within each project, allocate across directories proportionally
      3. Within each directory, pick by complexity spread
    """
    if len(paths) <= n:
        return paths

    # Level 1: group by project
    project_groups = {}  # type: Dict[str, List[str]]
    for p in paths:
        proj = file_to_project.get(p, "_unknown") if file_to_project else "_all"
        project_groups.setdefault(proj, []).append(p)

    # Allocate slots to projects (proportional, min 1 each)
    proj_allocation = _proportional_allocate(
        {proj: len(members) for proj, members in project_groups.items()},
        n,
    )

    selected = []
    for proj, proj_slots in proj_allocation.items():
        members = project_groups[proj]
        if proj_slots <= 0:
            continue

        # Level 2: within this project, group by directory
        dir_groups = {}  # type: Dict[str, List[str]]
        for p in members:
            d = str(Path(p).parent)
            dir_groups.setdefault(d, []).append(p)

        dir_allocation = _proportional_allocate(
            {d: len(ms) for d, ms in dir_groups.items()},
            proj_slots,
        )

        # Level 3: pick by complexity within each directory
        for d, dir_slots in dir_allocation.items():
            if dir_slots <= 0:
                continue
            dir_members = dir_groups[d]
            by_size = sorted(dir_members, key=lambda p: len(skeletons.get(p, "")))
            if len(by_size) <= dir_slots:
                selected.extend(by_size)
            else:
                step = len(by_size) / dir_slots
                selected.extend(by_size[int(i * step)] for i in range(dir_slots))

    # Fill any remaining slots
    if len(selected) < n:
        selected_set = set(selected)
        remaining = [p for p in paths if p not in selected_set]
        random.shuffle(remaining)
        selected.extend(remaining[: n - len(selected)])

    return selected[:n]


def _recent_sample(paths: List[str], n: int) -> List[str]:
    """
    Select the *n* most recently modified files (by mtime, descending).
    Falls back gracefully if a file has been removed between discovery and
    sampling — missing files sort to the bottom.
    """
    if len(paths) <= n:
        return paths

    def _mtime(p: str) -> float:
        try:
            return os.path.getmtime(p)
        except OSError:
            return 0.0

    by_mtime = sorted(paths, key=_mtime, reverse=True)
    return by_mtime[:n]


def _proportional_allocate(group_sizes: Dict[str, int], total_slots: int) -> Dict[str, int]:
    """
    Allocate total_slots across groups proportionally to size.
    Each group gets at least 1 slot (if slots permit).
    """
    total = sum(group_sizes.values())
    if total == 0:
        return {}

    groups = sorted(group_sizes.keys(), key=lambda g: -group_sizes[g])
    allocation = {}
    remaining = total_slots

    # First: 1 slot per group
    for g in groups:
        if remaining <= 0:
            break
        allocation[g] = 1
        remaining -= 1

    # Then: proportional extras
    if remaining > 0:
        for g in groups:
            if g not in allocation:
                continue
            extra = int((group_sizes[g] / total) * remaining)
            allocation[g] += extra

    # Distribute any leftover to largest groups
    allocated = sum(allocation.values())
    leftover = total_slots - allocated
    for g in groups:
        if leftover <= 0:
            break
        if g in allocation:
            allocation[g] += 1
            leftover -= 1

    return allocation


def _analyze_single_file(
    p: str,
    cat: str,
    cat_dir: Path,
    skeletons: dict,
    file_to_project: dict,
    category_label: str,
    index: int,
    total: int,
    t0: float,
) -> Tuple[str, str]:
    """Analyze a single file (called from thread pool). Returns (path, analysis_text)."""
    cache_file = cat_dir / (_cache_key(p) + ".analysis.md")
    if cache_file.exists():
        with _print_lock:
            print(f"  [{cat}] ({index}/{total}) cached: {Path(p).name}")
        return p, cache_file.read_text()

    project_name = file_to_project.get(p, "") if file_to_project else ""
    with _print_lock:
        print(f"  [{cat}] ({index}/{total}) analyzing: {Path(p).name} [project={project_name}] ({_elapsed(t0)})")
    skeleton = skeletons[cat][p]

    # Read raw source, truncated to fit comfortably in context
    try:
        raw = Path(p).read_text(errors="replace")
        raw = _truncate_source(raw, int(_char_budget() * 0.55))
    except Exception:
        raw = "[could not read file]"

    prompt = build_analysis_prompt(
        category=category_label,
        file_path=p,
        skeleton=skeleton,
        raw_source=raw,
        project_name=project_name,
    )

    result = call_llm(prompt)
    cache_file.write_text(result)
    return p, result


def phase_analyze(
    sampled: dict,
    skeletons: dict,
    work_dir: Path,
    file_to_project: dict = None,
) -> dict:
    """Phase 2: per-file LLM analysis. Cached to disk for resumability."""
    analyses = {}
    for cat, paths in sampled.items():
        cat_dir = work_dir / cat / "analyses"
        cat_dir.mkdir(parents=True, exist_ok=True)
        analyses[cat] = {}
        category_label = CATEGORIES[cat]["label"]

        t0 = time.time()

        if _max_workers <= 1:
            # Sequential path — same behaviour as before
            for i, p in enumerate(paths, 1):
                _, result = _analyze_single_file(
                    p, cat, cat_dir, skeletons,
                    file_to_project or {}, category_label,
                    i, len(paths), t0,
                )
                analyses[cat][p] = result
        else:
            # Parallel path
            with ThreadPoolExecutor(max_workers=_max_workers) as pool:
                futures = {
                    pool.submit(
                        _analyze_single_file,
                        p, cat, cat_dir, skeletons,
                        file_to_project or {}, category_label,
                        i, len(paths), t0,
                    ): p
                    for i, p in enumerate(paths, 1)
                }
                for future in as_completed(futures):
                    filepath, result = future.result()
                    analyses[cat][filepath] = result

    return analyses


def _hierarchical_reduce(
    chunks: List[str],
    reduce_prompt_fn,
    label: str,
    work_dir: Path,
    cache_prefix: str,
) -> str:
    """
    Recursively merge a list of text chunks until they fit in a single prompt.

    reduce_prompt_fn(texts: list[str]) -> str  — builds the merge prompt.

    At each level, groups chunks into batches that fit in the context budget,
    asks the LLM to summarize each batch, and repeats with the summaries
    until only one remains.
    """
    level = 0
    current = chunks

    while len(current) > 1:
        prev_count = len(current)
        level += 1
        cache_dir = work_dir / f"{cache_prefix}_level{level}"
        cache_dir.mkdir(parents=True, exist_ok=True)

        # Determine batch size: how many chunks fit in a single prompt?
        budget = _char_budget()
        batches = _pack_batches(current, budget)

        print(f"    [{label}] reduce level {level}: {len(current)} chunks -> {len(batches)} batches")

        next_level = [None] * len(batches)

        def _reduce_batch(bi, batch):
            cache_file = cache_dir / f"batch_{bi}.md"
            if cache_file.exists():
                return bi, cache_file.read_text()
            if len(batch) == 1:
                cache_file.write_text(batch[0])
                return bi, batch[0]
            prompt = reduce_prompt_fn(batch)
            result = call_llm(prompt)
            cache_file.write_text(result)
            return bi, result

        if _max_workers <= 1 or len(batches) <= 1:
            for bi, batch in enumerate(batches):
                _, result = _reduce_batch(bi, batch)
                next_level[bi] = result
        else:
            with ThreadPoolExecutor(max_workers=_max_workers) as pool:
                futures = [
                    pool.submit(_reduce_batch, bi, batch)
                    for bi, batch in enumerate(batches)
                ]
                for future in as_completed(futures):
                    bi, result = future.result()
                    next_level[bi] = result

        current = next_level

        # Safety valve: if we're not making progress, break
        if len(current) >= prev_count:
            print(f"    [{label}] WARNING: reduce not converging, stopping at {len(current)} chunks")
            break

    return current[0] if current else ""


def _deduplicate_chunks(chunks: List[str], similarity_threshold: float = 0.85) -> List[str]:
    """
    Deterministically remove near-duplicate chunks before LLM reduction.

    Uses line-level set similarity (Jaccard index) to identify chunks that are
    essentially the same content. When duplicates are found, keeps one representative
    and appends a count note. This lets the LLM spend its capacity on genuinely
    distinct patterns rather than re-discovering duplicates.
    """
    if len(chunks) <= 1:
        return chunks

    def _line_set(text):
        """Normalize and extract meaningful lines as a set for comparison."""
        lines = set()
        for line in text.split("\n"):
            stripped = line.strip()
            # Skip empty lines, comments, and very short lines
            if stripped and len(stripped) > 5:
                lines.add(stripped.lower())
        return lines

    def _jaccard(s1, s2):
        if not s1 or not s2:
            return 0.0
        intersection = len(s1 & s2)
        union = len(s1 | s2)
        return intersection / union if union > 0 else 0.0

    line_sets = [_line_set(c) for c in chunks]
    merged = [False] * len(chunks)
    merge_counts = [1] * len(chunks)

    # Greedily merge duplicates: for each chunk, check if it's similar to
    # an earlier non-merged chunk
    for i in range(1, len(chunks)):
        if merged[i]:
            continue
        for j in range(i):
            if merged[j]:
                continue
            if _jaccard(line_sets[i], line_sets[j]) >= similarity_threshold:
                merged[i] = True
                merge_counts[j] += 1
                break

    result = []
    for i, chunk in enumerate(chunks):
        if merged[i]:
            continue
        if merge_counts[i] > 1:
            result.append(f"{chunk}\n\n[NOTE: {merge_counts[i]} near-identical files exhibited this pattern]")
        else:
            result.append(chunk)

    deduped = len(chunks) - len(result)
    if deduped > 0:
        print(f"      dedup: {len(chunks)} chunks -> {len(result)} ({deduped} near-duplicates merged)")

    return result


def _pack_batches(chunks: List[str], char_budget: int) -> List[List[str]]:
    """
    Pack chunks into batches where each batch's total length fits in char_budget.
    Greedy first-fit. A single chunk larger than the budget gets its own batch.
    """
    batches = []  # type: List[List[str]]
    current_batch = []  # type: List[str]
    current_size = 0

    for chunk in chunks:
        chunk_len = len(chunk)

        if current_batch and current_size + chunk_len > char_budget:
            batches.append(current_batch)
            current_batch = []
            current_size = 0

        current_batch.append(chunk)
        current_size += chunk_len

    if current_batch:
        batches.append(current_batch)

    return batches


def _extract_structured_data(analysis_text: str) -> Optional[dict]:
    """Extract the STRUCTURED_DATA JSON block from an analysis text."""
    import re as _re
    match = _re.search(
        r'```(?:json)?\s*STRUCTURED_DATA\s*\n(.*?)```',
        analysis_text,
        _re.DOTALL,
    )
    if not match:
        return None
    try:
        return json.loads(match.group(1).strip())
    except (json.JSONDecodeError, ValueError):
        return None


def _merge_structured_data(data_list: List[dict]) -> str:
    """
    Deterministically merge structured JSON data from multiple file analyses.

    For each field:
    - Strings: count occurrences, report dominant + variants
    - Arrays: union all values with frequency counts
    - Booleans: count True vs False
    - Numbers: report min/median/max/mode
    """
    if not data_list:
        return ""

    lines = []

    def _merge_string_field(label, values):
        """Summarize a string field across files."""
        counts = Counter(v for v in values if v is not None)
        if not counts:
            return
        lines.append(f"\n{label}:")
        for val, freq in counts.most_common():
            lines.append(f"  {val}: {freq} files")

    def _merge_array_field(label, arrays):
        """Union arrays across files with frequency counts."""
        counts = Counter()
        for arr in arrays:
            if isinstance(arr, list):
                for item in arr:
                    counts[item] += 1
        if not counts:
            return
        lines.append(f"\n{label}:")
        for val, freq in counts.most_common():
            lines.append(f"  {val}: {freq} occurrences")

    def _merge_bool_field(label, values):
        """Count True vs False."""
        t = sum(1 for v in values if v is True)
        f = sum(1 for v in values if v is False)
        if t + f == 0:
            return
        lines.append(f"\n{label}: True={t}, False={f}")

    def _merge_number_field(label, values):
        """Report mode for number fields."""
        nums = [v for v in values if isinstance(v, (int, float))]
        if not nums:
            return
        counts = Counter(nums)
        mode_val, mode_freq = counts.most_common(1)[0]
        lines.append(f"\n{label}: mode={mode_val} ({mode_freq} files), range=[{min(nums)}, {max(nums)}]")

    # Naming section
    lines.append("=== Naming (deterministic merge) ===")
    _merge_string_field("Class casing", [d.get("naming", {}).get("class_casing") for d in data_list])
    _merge_string_field("Class suffix", [d.get("naming", {}).get("class_suffix") for d in data_list])
    _merge_string_field("Method casing", [d.get("naming", {}).get("method_casing") for d in data_list])
    _merge_array_field("Method prefixes", [d.get("naming", {}).get("method_prefixes", []) for d in data_list])
    _merge_string_field("Field casing", [d.get("naming", {}).get("field_casing") for d in data_list])
    _merge_string_field("Field prefix", [d.get("naming", {}).get("field_prefix") for d in data_list])
    _merge_array_field("Boolean naming", [d.get("naming", {}).get("boolean_naming", []) for d in data_list])
    _merge_string_field("Constant casing", [d.get("naming", {}).get("constant_casing") for d in data_list])
    _merge_string_field("Param casing", [d.get("naming", {}).get("param_casing") for d in data_list])

    # Ordering section
    lines.append("\n=== Ordering (deterministic merge) ===")
    _merge_array_field("Import groups", [d.get("ordering", {}).get("imports", []) for d in data_list])
    # For field/method ordering, show the full sequences with frequency
    field_orders = [
        " -> ".join(d.get("ordering", {}).get("fields", []))
        for d in data_list
        if d.get("ordering", {}).get("fields")
    ]
    if field_orders:
        _merge_string_field("Field ordering sequences", field_orders)
    method_orders = [
        " -> ".join(d.get("ordering", {}).get("methods", []))
        for d in data_list
        if d.get("ordering", {}).get("methods")
    ]
    if method_orders:
        _merge_string_field("Method ordering sequences", method_orders)

    # Annotations section
    lines.append("\n=== Annotations (deterministic merge) ===")
    _merge_array_field("Class-level annotations", [d.get("annotations", {}).get("class_level", []) for d in data_list])
    _merge_array_field("Field-level annotations", [d.get("annotations", {}).get("field_level", []) for d in data_list])
    _merge_array_field("Method-level annotations", [d.get("annotations", {}).get("method_level", []) for d in data_list])

    # Formatting section
    lines.append("\n=== Formatting (deterministic merge) ===")
    _merge_string_field("Brace style", [d.get("formatting", {}).get("brace_style") for d in data_list])
    _merge_string_field("Class brace style", [d.get("formatting", {}).get("class_brace_style") for d in data_list])
    _merge_bool_field("One-line getters", [d.get("formatting", {}).get("one_line_getters") for d in data_list])
    _merge_bool_field("One-line setters", [d.get("formatting", {}).get("one_line_setters") for d in data_list])
    _merge_number_field("Blank lines between fields", [d.get("formatting", {}).get("blank_lines_between_fields") for d in data_list])
    _merge_number_field("Blank lines between methods", [d.get("formatting", {}).get("blank_lines_between_methods") for d in data_list])
    _merge_bool_field("Getter/setter adjacent", [d.get("formatting", {}).get("getter_setter_adjacent") for d in data_list])
    _merge_bool_field("Single-statement braces", [d.get("formatting", {}).get("single_statement_braces") for d in data_list])

    # Patterns section
    lines.append("\n=== Patterns (deterministic merge) ===")
    _merge_array_field("Design patterns", [d.get("patterns", {}).get("design_patterns", []) for d in data_list])
    _merge_string_field("Error handling", [d.get("patterns", {}).get("error_handling") for d in data_list])
    _merge_string_field("Null handling", [d.get("patterns", {}).get("null_handling") for d in data_list])
    _merge_string_field("Base class", [d.get("patterns", {}).get("base_class") for d in data_list])
    _merge_array_field("Interfaces", [d.get("patterns", {}).get("interfaces", []) for d in data_list])
    _merge_array_field("Standard methods", [d.get("patterns", {}).get("standard_methods", []) for d in data_list])

    return "\n".join(lines)


def _aggregate_skeleton_stats(skeletons_for_cat: dict) -> str:
    """
    Deterministically extract and aggregate hard numbers from skeleton metadata.

    Parses the // [FORMATTING: ...], // [MEMBER ORDER: ...], and // [NAMING: ...]
    comments from all skeletons and produces aggregate statistics that can be
    passed directly into synthesis without LLM re-interpretation.
    """
    import re as _re

    formatting_totals = {
        "brace_same_line": 0,
        "brace_next_line": 0,
        "class_brace_same_line": 0,
        "class_brace_next_line": 0,
        "one_line_methods": 0,
        "multi_line_methods": 0,
        "one_line_getters": 0,
        "one_line_setters": 0,
        "getter_setter_pairs": 0,
    }
    member_spacing_counts = Counter()
    ordering_sequences = Counter()
    class_suffixes = Counter()
    field_naming_styles = Counter()
    method_prefix_counts = Counter()
    files_counted = 0

    for _path, skel in skeletons_for_cat.items():
        files_counted += 1

        # Parse FORMATTING line
        fmt_match = _re.search(r'// \[FORMATTING: (.*?)\]', skel)
        if fmt_match:
            fmt_str = fmt_match.group(1)

            # Brace placement
            m = _re.search(r'braces=same-line\b(?!\()', fmt_str)
            if m:
                formatting_totals["brace_same_line"] += 1
            m = _re.search(r'braces=next-line\b(?!\()', fmt_str)
            if m:
                formatting_totals["brace_next_line"] += 1
            m = _re.search(r'braces=same-line\((\d+)\)/next-line\((\d+)\)', fmt_str)
            if m:
                formatting_totals["brace_same_line"] += int(m.group(1))
                formatting_totals["brace_next_line"] += int(m.group(2))

            # Class brace
            if "class-brace=same-line" in fmt_str:
                formatting_totals["class_brace_same_line"] += 1
            elif "class-brace=next-line" in fmt_str:
                formatting_totals["class_brace_next_line"] += 1

            # One-line methods
            m = _re.search(r'one-line-methods=(\d+)', fmt_str)
            if m:
                formatting_totals["one_line_methods"] += int(m.group(1))
            m = _re.search(r'multi-line-methods=(\d+)', fmt_str)
            if m:
                formatting_totals["multi_line_methods"] += int(m.group(1))

            # One-line getters/setters
            m = _re.search(r'getters=(\d+)', fmt_str)
            if m:
                formatting_totals["one_line_getters"] += int(m.group(1))
            m = _re.search(r'setters=(\d+)', fmt_str)
            if m:
                formatting_totals["one_line_setters"] += int(m.group(1))

            # Getter-setter pairs
            m = _re.search(r'getter-setter-pairs=(\d+)', fmt_str)
            if m:
                formatting_totals["getter_setter_pairs"] += int(m.group(1))

            # Member spacing
            m = _re.search(r'member-spacing=(\d+)-blank-lines', fmt_str)
            if m:
                member_spacing_counts[int(m.group(1))] += 1
            else:
                for sm in _re.finditer(r'(\d+)bl\((\d+)\)', fmt_str):
                    member_spacing_counts[int(sm.group(1))] += int(sm.group(2))

        # Parse MEMBER ORDER line
        order_match = _re.search(r'// \[MEMBER ORDER: (.*?)\]', skel)
        if order_match:
            ordering_sequences[order_match.group(1).strip()] += 1

        # Parse NAMING line
        naming_match = _re.search(r'// \[NAMING: (.*?)\]', skel)
        if naming_match:
            naming_str = naming_match.group(1)
            # Class suffix
            m = _re.search(r'class suffix=(\w+)', naming_str)
            if m:
                class_suffixes[m.group(1)] += 1
            # Field naming style
            m = _re.search(r'fields=(\w+)', naming_str)
            if m:
                field_naming_styles[m.group(1)] += 1

        # Parse METHOD NAMING line
        method_match = _re.search(r'// \[METHOD NAMING: (.*?)\]', skel)
        if method_match:
            for mp in _re.finditer(r'(\w+)\*\((\d+)\)', method_match.group(1)):
                method_prefix_counts[mp.group(1)] += int(mp.group(2))

    if files_counted == 0:
        return ""

    # Build output
    lines = [f"Total files with skeletons: {files_counted}"]

    # Brace placement
    same = formatting_totals["brace_same_line"]
    nxt = formatting_totals["brace_next_line"]
    if same + nxt > 0:
        lines.append(f"\nBrace Placement (method/control-flow):")
        lines.append(f"  Same line (K&R): {same} occurrences")
        lines.append(f"  Next line (Allman): {nxt} occurrences")
        pct = same / (same + nxt) * 100
        lines.append(f"  -> {pct:.0f}% same-line")

    csame = formatting_totals["class_brace_same_line"]
    cnxt = formatting_totals["class_brace_next_line"]
    if csame + cnxt > 0:
        lines.append(f"\nClass Brace Placement:")
        lines.append(f"  Same line: {csame} files")
        lines.append(f"  Next line: {cnxt} files")

    # One-line methods
    one = formatting_totals["one_line_methods"]
    multi = formatting_totals["multi_line_methods"]
    if one + multi > 0:
        lines.append(f"\nMethod Body Style:")
        lines.append(f"  One-line methods: {one} (getters={formatting_totals['one_line_getters']}, setters={formatting_totals['one_line_setters']})")
        lines.append(f"  Multi-line methods: {multi}")

    # Getter-setter proximity
    gs = formatting_totals["getter_setter_pairs"]
    if gs > 0:
        lines.append(f"\nGetter/Setter Proximity:")
        lines.append(f"  Adjacent getter-setter pairs: {gs}")

    # Member spacing
    if member_spacing_counts:
        lines.append(f"\nMember Spacing (blank lines between members):")
        for count, freq in member_spacing_counts.most_common():
            lines.append(f"  {count} blank line(s): {freq} occurrences")

    # Member ordering
    if ordering_sequences:
        lines.append(f"\nMember Ordering Patterns (top {min(5, len(ordering_sequences))}):")
        for seq, freq in ordering_sequences.most_common(5):
            lines.append(f"  [{freq}x] {seq}")

    # Class suffixes
    if class_suffixes:
        lines.append(f"\nClass Name Suffixes:")
        for suffix, freq in class_suffixes.most_common():
            lines.append(f"  *{suffix}: {freq} files")

    # Field naming
    if field_naming_styles:
        lines.append(f"\nField Naming Styles:")
        for style, freq in field_naming_styles.most_common():
            lines.append(f"  {style}: {freq} files")

    # Method prefixes
    if method_prefix_counts:
        lines.append(f"\nMethod Verb Prefixes (across all files):")
        for prefix, freq in method_prefix_counts.most_common(15):
            lines.append(f"  {prefix}*: {freq} methods")

    return "\n".join(lines)


def phase_skeleton_overview(
    skeletons: dict,
    sampled: dict,
    work_dir: Path,
) -> dict:
    """
    Phase 2b: scan ALL non-sampled skeletons for broad pattern coverage.

    Uses hierarchical reduce:
      1. Group skeletons into batches that fit in context
      2. LLM summarizes patterns in each batch
      3. Merge summaries until one overview remains
    """
    overviews = {}
    for cat, all_skels in skeletons.items():
        cat_dir = work_dir / cat
        cache_file = cat_dir / "skeleton_overview.md"
        if cache_file.exists():
            overviews[cat] = cache_file.read_text()
            print(f"  [{cat}] skeleton overview: cached")
            continue

        sampled_set = set(sampled.get(cat, []))
        remaining = {p: s for p, s in all_skels.items() if p not in sampled_set}

        if not remaining:
            overviews[cat] = ""
            continue

        category_label = CATEGORIES[cat]["label"]
        print(f"  [{cat}] scanning {len(remaining)} skeletons ...")

        # Build chunks: each chunk is a group of skeletons formatted together
        # Pre-group by directory for locality
        by_dir = {}  # type: Dict[str, List[Tuple[str, str]]]
        for p, s in remaining.items():
            d = str(Path(p).parent)
            by_dir.setdefault(d, []).append((p, s))

        # Flatten into formatted skeleton chunks, one per directory group
        formatted_chunks = []
        for d, items in sorted(by_dir.items()):
            text = "\n\n".join(
                f"### {Path(p).name}\n{skel}" for p, skel in items
            )
            # If a single directory has too many files, split it
            if len(text) > _char_budget():
                sub_chunks = [
                    f"### {Path(p).name}\n{skel}" for p, skel in items
                ]
                formatted_chunks.extend(sub_chunks)
            else:
                formatted_chunks.append(text)

        def skeleton_reduce_prompt(texts):
            # type: (List[str]) -> str
            combined = "\n\n---\n\n".join(texts)
            return (
                f"Below are structural skeletons of {category_label} files from a codebase.\n"
                f"Identify high-level patterns: naming conventions, common annotations, "
                f"typical class structures, recurring method signatures, inheritance patterns, "
                f"and any notable conventions.\n"
                f"Note frequency for each pattern (how many files exhibited it).\n"
                f"RARE PATTERNS: Patterns observed in only 1-2 files MUST be preserved "
                f"in a separate '### Rare/Notable Patterns' section. Do NOT discard them.\n"
                f"Be thorough — output a structured list of patterns.\n\n"
                f"{combined}"
            )

        # Deduplicate near-identical skeleton chunks before LLM reduction (change C)
        formatted_chunks = _deduplicate_chunks(formatted_chunks)

        overview = _hierarchical_reduce(
            formatted_chunks,
            skeleton_reduce_prompt,
            label=cat,
            work_dir=cat_dir,
            cache_prefix="skel_reduce",
        )

        cache_file.write_text(overview)
        overviews[cat] = overview

    return overviews


def phase_synthesize(
    analyses: dict,
    skeleton_overviews: dict,
    work_dir: Path,
    file_to_project: dict = None,
    skeletons: dict = None,
) -> dict:
    """
    Phase 3: merge all analyses into a single SKILL.md per category.

    Each analysis chunk is tagged with its project name so the LLM can identify
    project-specific patterns vs universal patterns.
    """
    skills = {}
    for cat, file_analyses in analyses.items():
        cache_file = work_dir / cat / "SKILL_draft.md"
        if cache_file.exists():
            print(f"  [{cat}] synthesis: cached")
            skills[cat] = cache_file.read_text()
            continue

        category_label = CATEGORIES[cat]["label"]
        cat_dir = work_dir / cat

        # Format each analysis with project tag
        analysis_chunks = []
        for p, text in file_analyses.items():
            proj = file_to_project.get(p, "unknown") if file_to_project else "unknown"
            analysis_chunks.append(
                f"### {Path(p).name}  [project: {proj}]\n{text}"
            )

        # Extract and merge structured JSON data from analyses (change B)
        structured_data_list = []
        for p, text in file_analyses.items():
            sd = _extract_structured_data(text)
            if sd:
                structured_data_list.append(sd)
        merged_structured = ""
        if structured_data_list:
            merged_structured = _merge_structured_data(structured_data_list)
            merge_cache = cat_dir / "merged_structured_data.txt"
            merge_cache.write_text(merged_structured)
            print(f"    [{cat}] extracted structured data from {len(structured_data_list)}/{len(file_analyses)} analyses")

        # Build project distribution stats
        project_stats = ""
        if file_to_project:
            proj_counts = Counter(
                file_to_project.get(p, "unknown") for p in file_analyses.keys()
            )
            lines = [f"Files analyzed per project:"]
            for proj, count in proj_counts.most_common():
                lines.append(f"  - {proj}: {count} files")
            project_stats = "\n".join(lines)

        skel_overview = skeleton_overviews.get(cat, "")

        # Aggregate deterministic stats from ALL skeletons (change D)
        # and merged structured analysis data (change B)
        skel_stats_parts = []
        if skeletons and cat in skeletons:
            skel_stats_parts.append(_aggregate_skeleton_stats(skeletons[cat]))
        if merged_structured:
            skel_stats_parts.append(merged_structured)
        skel_stats = "\n\n".join(p for p in skel_stats_parts if p)

        # Check if everything fits in one prompt
        total_analysis_chars = sum(len(c) for c in analysis_chunks)
        total_chars = total_analysis_chars + len(skel_overview) + len(project_stats) + 3000

        if total_chars <= _char_budget():
            print(f"  [{cat}] synthesizing SKILL.md (single pass) ...")
            all_text = "\n\n---\n\n".join(analysis_chunks)
            prompt = build_synthesis_prompt(
                category=category_label,
                analyses_text=all_text,
                skeleton_overview=skel_overview,
                project_stats=project_stats,
                skeleton_stats=skel_stats,
                max_skill_lines=_max_skill_lines,
            )
            result = call_llm(prompt)
        else:
            print(f"  [{cat}] synthesizing SKILL.md (hierarchical, {len(analysis_chunks)} chunks) ...")

            def analysis_reduce_prompt(texts):
                # type: (List[str]) -> str
                combined = "\n\n---\n\n".join(texts)
                return (
                    f"Below are pattern analyses of multiple {category_label} files "
                    f"from a codebase. Each analysis is tagged with its project name "
                    f"(e.g. [project: foo]).\n\n"
                    f"Merge them into a consolidated list of patterns, conventions, "
                    f"and best practices.\n"
                    f"- Combine duplicate observations\n"
                    f"- Note frequency (how many files exhibited each pattern)\n"
                    f"- PRESERVE PROJECT TAGS: note which patterns are universal "
                    f"vs specific to certain projects\n"
                    f"- Preserve specific examples (annotation names, method signatures, "
                    f"field names, naming conventions, ordering conventions)\n"
                    f"- Flag any contradictions or variations between projects\n"
                    f"- RARE PATTERNS: Patterns observed in only 1-2 files MUST be preserved "
                    f"in a separate '### Rare/Notable Patterns' section at the end. "
                    f"Do NOT merge these with common patterns — they may represent "
                    f"important project-specific conventions or edge cases.\n"
                    f"Be thorough — prefer completeness over brevity.\n\n{combined}"
                )

            # Deduplicate near-identical analyses before LLM reduction (change C)
            deduped_chunks = _deduplicate_chunks(analysis_chunks)

            condensed = _hierarchical_reduce(
                deduped_chunks,
                analysis_reduce_prompt,
                label=cat,
                work_dir=cat_dir,
                cache_prefix="analysis_reduce",
            )

            # Reduce skeleton overview if needed
            remaining_budget = _char_budget() - len(condensed) - len(project_stats) - 3000
            if skel_overview and len(skel_overview) > remaining_budget:
                print(f"    [{cat}] also reducing skeleton overview ...")
                paras = [p for p in skel_overview.split("\n\n") if p.strip()]
                if paras:
                    def skel_merge_prompt(texts):
                        # type: (List[str]) -> str
                        combined = "\n\n".join(texts)
                        return (
                            f"Consolidate these pattern observations about {category_label} "
                            f"files into a summary. Remove duplicates, keep specific "
                            f"examples. Preserve project-specific notes.\n"
                            f"RARE PATTERNS: Keep patterns observed in only 1-2 files in a "
                            f"separate '### Rare/Notable Patterns' section.\n\n{combined}"
                        )
                    skel_overview = _hierarchical_reduce(
                        paras,
                        skel_merge_prompt,
                        label=f"{cat}-skel",
                        work_dir=cat_dir,
                        cache_prefix="skel_overview_reduce",
                    )

            prompt = build_synthesis_prompt(
                category=category_label,
                analyses_text=condensed,
                skeleton_overview=skel_overview,
                project_stats=project_stats,
                skeleton_stats=skel_stats,
                max_skill_lines=_max_skill_lines,
            )
            result = call_llm(prompt)

        cache_file.write_text(result)
        skills[cat] = result

    return skills


def phase_validate(
    skills: dict,
    skeletons: dict,
    sampled: dict,
    work_dir: Path,
    file_to_project: dict = None,
) -> dict:
    """Phase 4 (optional): validate the SKILL.md against unseen files."""
    reports = {}
    for cat, skill_text in skills.items():
        cache_file = work_dir / cat / "validation_report.md"
        if cache_file.exists():
            reports[cat] = cache_file.read_text()
            print(f"  [{cat}] validation: cached")
            continue

        sampled_set = set(sampled.get(cat, []))
        unseen = [p for p in skeletons.get(cat, {}) if p not in sampled_set]
        if not unseen:
            reports[cat] = "No unseen files available for validation."
            continue

        # Stratified by directory, pick from different projects
        by_dir = {}  # type: Dict[str, List[str]]
        for p in unseen:
            d = str(Path(p).parent)
            by_dir.setdefault(d, []).append(p)
        test_files = []
        dirs = list(by_dir.keys())
        random.shuffle(dirs)
        for d in dirs:
            if len(test_files) >= 5:
                break
            test_files.append(random.choice(by_dir[d]))

        category_label = CATEGORIES[cat]["label"]

        # Build project info for test files
        test_projects = ""
        if file_to_project:
            projs = set(file_to_project.get(p, "unknown") for p in test_files)
            test_projects = ", ".join(sorted(projs))

        file_budget = (_char_budget() - len(skill_text) - 1000) // len(test_files)
        file_sections = ""
        for p in test_files:
            proj = file_to_project.get(p, "unknown") if file_to_project else ""
            proj_tag = f"  [project: {proj}]" if proj else ""
            try:
                raw = Path(p).read_text(errors="replace")
                raw = _truncate_source(raw, file_budget)
            except Exception:
                raw = skeletons[cat].get(p, "[could not read]")

            file_sections += f"\n\n### {Path(p).name}{proj_tag}\n```\n{raw}\n```\n"

        prompt = build_validation_prompt(
            category=category_label,
            skill_md=skill_text,
            test_files=file_sections,
            test_file_projects=test_projects,
        )

        print(f"  [{cat}] validating against {len(test_files)} unseen files ...")
        result = call_llm(prompt)
        cache_file.write_text(result)
        reports[cat] = result

    return reports


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def write_final_skills(
    skills: dict,
    validation_reports: dict,
    output_dir: Path,
):
    """Write the final SKILL.md files, optionally incorporating validation feedback."""
    output_dir.mkdir(parents=True, exist_ok=True)

    for cat, skill_text in skills.items():
        out_path = output_dir / cat
        out_path.mkdir(parents=True, exist_ok=True)

        skill_file = out_path / "SKILL.md"
        skill_file.write_text(skill_text)
        print(f"  -> {skill_file}")

        # Also write validation report alongside if it exists
        report = validation_reports.get(cat, "")
        if report:
            (out_path / "VALIDATION_REPORT.md").write_text(report)

    print(f"\nDone! Skills written to {output_dir}/")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Analyze a codebase and generate SKILL.md files for agentic coding.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
examples:
  %(prog)s /src/myproject
  %(prog)s /src/myproject --model llama3.1
  %(prog)s /src/myproject --api-base http://gpu-box:8000 --model meta-llama/Llama-3.1-8B
  %(prog)s /src/myproject --api-base http://localhost:1234 --api-type openai
  %(prog)s /src/myproject --api-key tok_abc123
  %(prog)s /src/myproject --header "X-API-Key: my-secret-token"
  %(prog)s /src/myproject --header "X-Custom: foo" --header "X-Other: bar"

LLM server compatibility:
  Ollama (default)     --api-base http://localhost:11434
  vLLM                 --api-base http://host:8000 --api-type openai
  SGLang               --api-base http://host:30000 --api-type openai
  LM Studio            --api-base http://localhost:1234 --api-type openai
  llama.cpp            --api-base http://localhost:8080 --api-type openai
  Any OpenAI-compat    --api-base http://host:port --api-type openai

  No external Python packages required — uses only the standard library.
        """,
    )
    parser.add_argument("root_dir", help="Root directory of the codebase to analyze")
    parser.add_argument(
        "--model",
        default="llama3.1",
        help="Model name as your LLM server knows it (default: llama3.1)",
    )
    parser.add_argument(
        "--api-base",
        default="https://api.openai.com",
        help="Base URL of your LLM server (default: https://api.openai.com). "
             "For vLLM/SGLang pass just host:port — /v1 path is added automatically.",
    )
    parser.add_argument(
        "--api-type",
        choices=["auto", "ollama", "openai"],
        default="auto",
        help="API protocol: 'ollama' for Ollama's native API, 'openai' for "
             "OpenAI-compatible (vLLM, SGLang, LM Studio, llama.cpp). "
             "'auto' detects from --api-base. (default: auto)",
    )
    parser.add_argument(
        "--api-key",
        default="",
        help="Bearer token sent as 'Authorization: Bearer <key>'. "
             "Falls back to OPENAI_API_KEY env var if not provided.",
    )
    parser.add_argument(
        "--header",
        action="append",
        default=[],
        metavar="'Name: value'",
        help="Extra HTTP header(s) to send with every LLM request. "
             "Repeatable. Example: --header 'X-API-Key: tok_abc'",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=20,
        help="Number of files to deeply analyze per category (default: 20). "
             "Larger = better coverage but more LLM calls.",
    )
    parser.add_argument(
        "--output-dir",
        default="./generated-skills",
        help="Where to write final SKILL.md files (default: ./generated-skills)",
    )
    parser.add_argument(
        "--work-dir",
        default="./.skill-gen-work",
        help="Working directory for intermediate results (default: ./.skill-gen-work)",
    )
    parser.add_argument(
        "--categories",
        nargs="+",
        choices=list(CATEGORIES.keys()),
        default=list(CATEGORIES.keys()),
        help="Which categories to process (default: all)",
    )
    parser.add_argument(
        "--projects",
        nargs="+",
        default=None,
        metavar="NAME",
        help="Only process these projects (immediate subdirectory names). "
             "By default all projects are included.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from cached intermediate results",
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Run validation phase against unseen files",
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Remove all cached intermediate results and start fresh",
    )
    parser.add_argument(
        "--skip-skeleton-overview",
        action="store_true",
        help="Skip the broad skeleton overview phase (faster, less thorough)",
    )
    parser.add_argument(
        "--context-budget",
        type=int,
        default=100_000,
        help="Your LLM's context window size in tokens. Used to size batches "
             "so prompts fit in context. (default: 100000)",
    )
    parser.add_argument(
        "--parallel",
        type=int,
        default=4,
        help="Max concurrent LLM requests. Set to 1 to disable parallelism. "
             "(default: 4)",
    )
    parser.add_argument(
        "--max-skill-lines",
        type=int,
        default=800,
        help="Maximum lines for the generated SKILL.md document. Higher values "
             "retain more detail at the cost of a longer document. (default: 800)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible sampling (default: 42)",
    )
    parser.add_argument(
        "--recent-files",
        action="store_true",
        help="Sample the N most recently modified files (by mtime) instead of "
             "using stratified random sampling. N is controlled by --sample-size.",
    )

    args = parser.parse_args()
    random.seed(args.seed)

    # Set the global token budget for batch sizing
    global _token_budget, _max_workers, _max_skill_lines
    _token_budget = args.context_budget
    _max_workers = max(1, args.parallel)
    _max_skill_lines = args.max_skill_lines

    # Parse --header flags into a dict
    extra_headers = {}
    for h in args.header:
        if ":" not in h:
            print(f"Error: invalid header format '{h}' — expected 'Name: value'", file=sys.stderr)
            sys.exit(1)
        name, _, value = h.partition(":")
        extra_headers[name.strip()] = value.strip()

    # Configure LLM connection
    configure_llm(
        api_base=args.api_base,
        model=args.model,
        api_type=args.api_type,
        api_key=args.api_key,
        extra_headers=extra_headers,
    )

    root = Path(args.root_dir).resolve()
    if not root.is_dir():
        print(f"Error: {root} is not a directory", file=sys.stderr)
        sys.exit(1)

    work_dir = Path(args.work_dir).resolve()
    output_dir = Path(args.output_dir).resolve()

    if args.clean and work_dir.exists():
        shutil.rmtree(work_dir)
        print(f"Cleaned {work_dir}")

    if not args.resume and work_dir.exists():
        print(
            f"Warning: {work_dir} exists. Use --resume to reuse cached results, "
            f"or --clean to start fresh.",
            file=sys.stderr,
        )

    work_dir.mkdir(parents=True, exist_ok=True)

    # === Pipeline ===
    print(f"\n{'='*60}")
    print(f" Codebase Skill Generator")
    print(f" root:       {root}")
    print(f" api:        {args.api_base}")
    print(f" model:      {args.model}")
    print(f" context:    {args.context_budget:,} tokens (~{_char_budget():,} char budget)")
    print(f" categories: {', '.join(args.categories)}")
    if args.projects:
        print(f" projects:   {', '.join(args.projects)}")
    print(f" parallel:   {_max_workers} worker(s)")
    sampling_label = "most-recent (mtime)" if args.recent_files else f"stratified (seed={args.seed})"
    print(f" sample:     {args.sample_size} files/category ({sampling_label})")
    print(f"{'='*60}\n")

    print("[Phase 0] Discovering files ...")
    files_by_cat, projects, file_to_project = phase_discover(
        str(root), args.categories, project_whitelist=args.projects,
    )
    for cat, paths in files_by_cat.items():
        # Report per-project breakdown
        proj_counts = Counter(file_to_project.get(p, "?") for p in paths)
        proj_summary = ", ".join(f"{proj}({n})" for proj, n in proj_counts.most_common(5))
        if len(proj_counts) > 5:
            proj_summary += f", ... +{len(proj_counts) - 5} more"
        print(f"  {cat}: {len(paths)} files [{proj_summary}]")
    if projects:
        print(f"  Projects detected: {', '.join(projects[:20])}")
        if len(projects) > 20:
            print(f"    ... and {len(projects) - 20} more")
    if not any(files_by_cat.values()):
        print("No files found! Check your root_dir and glob patterns.", file=sys.stderr)
        sys.exit(1)

    # Save file lists and project mapping
    for cat, paths in files_by_cat.items():
        cat_work = work_dir / cat
        cat_work.mkdir(parents=True, exist_ok=True)
        (cat_work / "files.json").write_text(json.dumps(paths, indent=2))
    (work_dir / "projects.json").write_text(json.dumps(projects, indent=2))
    (work_dir / "file_to_project.json").write_text(json.dumps(file_to_project, indent=2))

    print("\n[Phase 1] Extracting structural skeletons ...")
    skeletons = phase_extract_skeletons(files_by_cat, work_dir)
    for cat, s in skeletons.items():
        print(f"  {cat}: {len(s)} skeletons extracted")

    sampling_mode = "most-recent by mtime" if args.recent_files else "stratified random"
    print(f"\n[Phase 1b] Sampling representative files ({sampling_mode}) ...")
    sampled = phase_sample(
        files_by_cat, skeletons, args.sample_size, work_dir,
        file_to_project=file_to_project,
        recent=args.recent_files,
    )

    print("\n[Phase 2] Per-file LLM analysis ...")
    analyses = phase_analyze(sampled, skeletons, work_dir, file_to_project=file_to_project)

    skeleton_overviews = {}
    if not args.skip_skeleton_overview:
        print("\n[Phase 2b] Broad skeleton overview ...")
        skeleton_overviews = phase_skeleton_overview(
            skeletons, sampled, work_dir
        )

    print("\n[Phase 3] Synthesizing SKILL.md files ...")
    skills = phase_synthesize(
        analyses, skeleton_overviews, work_dir,
        file_to_project=file_to_project,
        skeletons=skeletons,
    )

    validation_reports = {}
    if args.validate:
        print("\n[Phase 4] Validating against unseen files ...")
        validation_reports = phase_validate(
            skills, skeletons, sampled, work_dir,
            file_to_project=file_to_project,
        )

    print("\n[Output] Writing final skills ...")
    write_final_skills(skills, validation_reports, output_dir)


if __name__ == "__main__":
    main()
