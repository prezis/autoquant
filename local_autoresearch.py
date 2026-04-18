#!/usr/bin/env python3
"""
local_autoresearch.py — Ollama-powered autonomous strategy optimization loop.

Karpathy autoresearch pattern:
  1. Read strategy.py + results.tsv + program.md
  2. Send to local Ollama (qwen3.5:27b) for proposed change
  3. Apply change to strategy.py via search/replace diffs
  4. Run backtest (python3 strategy.py)
  5. Parse score — keep if better, revert if worse
  6. Log experiment, repeat

Uses search/replace diffs instead of full file output — token-efficient
for large files (strategy.py is ~850 lines).

Runs entirely on local RTX 5090 via Ollama. Zero API cost.

Usage:
    python3 local_autoresearch.py                  # run loop (default 50 iterations)
    python3 local_autoresearch.py --max-iter 100   # custom iteration count
    python3 local_autoresearch.py --dry-run         # propose change but don't apply
"""

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

# ─── Configuration ───────────────────────────────────────────────

OLLAMA_URL = "http://localhost:11434/api/chat"
MODEL = "qwen3.5:27b"

PROJECT_DIR = Path(__file__).parent.resolve()
STRATEGY_FILE = PROJECT_DIR / "strategy.py"
RESULTS_FILE = PROJECT_DIR / "results.tsv"
PROGRAM_FILE = PROJECT_DIR / "program.md"
LOGS_DIR = PROJECT_DIR / "logi"
BACKUP_DIR = PROJECT_DIR / "logi" / "autoresearch_backups"
EXPERIMENT_LOG = PROJECT_DIR / "logi" / "autoresearch.jsonl"

# Backtest timeout (seconds) — LSTM training can take a while
BACKTEST_TIMEOUT = 900  # 15 min — BiLSTM/GRU h=384 3L x 300 epochs takes 8-10 min on RTX 5090

# How many recent results.tsv lines to include in prompt (keep context manageable)
MAX_RECENT_RESULTS = 30

# Max tokens for LLM generation (search/replace diffs are much shorter than full files)
MAX_PREDICT_TOKENS = 4096

# Patterns that must NOT appear in generated strategy code (safety check)
_BLOCKED_PATTERNS = [
    "subprocess.", "shutil.rmtree(", "rm -rf", "__import__",
    "os.system", "os.popen", "os.exec",
]


# ─── Ollama API ──────────────────────────────────────────────────

def ollama_wait_for_vram(min_free_mib: int = 20000, timeout_sec: int = 60):
    """Free GPU VRAM by stopping Ollama service before PyTorch training.

    IMPORTANT: keep_alive=0 and keep_alive="10s" do NOT work reliably because
    the Claude Code MCP local-ai server sends periodic requests to Ollama which
    reload the model within seconds. The ONLY reliable way to free VRAM is to
    stop the entire Ollama systemd service. systemctl stop is respected even
    with Restart=always (manual stops are not auto-restarted by systemd).
    DO NOT change this to use keep_alive — it has been tested and fails.
    """
    # Step 1: Stop Ollama service to guarantee VRAM release
    try:
        subprocess.run(["systemctl", "stop", "ollama"],
                       capture_output=True, timeout=15)
    except Exception as e:
        print(f"  Warning: systemctl stop ollama failed: {e}")

    # Step 2: Wait for VRAM to actually free
    for attempt in range(timeout_sec):
        try:
            nvsmi = subprocess.run(
                ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=5
            )
            free_mib = int(nvsmi.stdout.strip().split("\n")[0])
            if free_mib > min_free_mib:
                print(f"  Stopped Ollama, GPU VRAM free: {free_mib} MiB")
                return True
        except Exception:
            pass
        time.sleep(1)

    try:
        nvsmi = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5
        )
        free_mib = int(nvsmi.stdout.strip().split("\n")[0])
        print(f"  Warning: GPU VRAM only {free_mib} MiB free after {timeout_sec}s wait")
    except Exception:
        print("  Warning: could not check GPU VRAM")
    return False


def ollama_chat(messages: list[dict], temperature: float = 0.7,
                max_tokens: int = MAX_PREDICT_TOKENS) -> str:
    """Send chat request to Ollama and return assistant response text.
    Uses streaming to handle long generation times without timeout."""
    import urllib.request
    import urllib.error

    payload = json.dumps({
        "model": MODEL,
        "messages": messages,
        "stream": True,
        "keep_alive": "5m",  # default; VRAM freed by systemctl stop before backtest
        "options": {
            "temperature": temperature,
            "num_predict": max_tokens,
        },
    }).encode("utf-8")

    req = urllib.request.Request(
        OLLAMA_URL,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        response_text = []
        with urllib.request.urlopen(req, timeout=600) as resp:
            buffer = b""
            while True:
                chunk = resp.read(4096)
                if not chunk:
                    break
                buffer += chunk
                while b"\n" in buffer:
                    line, buffer = buffer.split(b"\n", 1)
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                        content = obj.get("message", {}).get("content", "")
                        if content:
                            response_text.append(content)
                        if obj.get("done", False):
                            break
                    except json.JSONDecodeError:
                        continue
            if buffer.strip():
                try:
                    obj = json.loads(buffer)
                    content = obj.get("message", {}).get("content", "")
                    if content:
                        response_text.append(content)
                except json.JSONDecodeError:
                    pass

        full_response = "".join(response_text)
        if not full_response.strip():
            raise RuntimeError("Ollama returned empty response")
        return full_response

    except urllib.error.URLError as e:
        raise RuntimeError(f"Cannot reach Ollama at {OLLAMA_URL}: {e}")
    except Exception as e:
        raise RuntimeError(f"Ollama request failed: {e}")


# ─── File helpers ────────────────────────────────────────────────

def read_file(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def write_file(path: Path, content: str):
    path.write_text(content, encoding="utf-8")


def backup_strategy(iteration: int) -> Path:
    """Create a backup of current strategy.py before modification."""
    BACKUP_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = BACKUP_DIR / f"strategy_iter{iteration:04d}_{ts}.py"
    shutil.copy2(STRATEGY_FILE, backup_path)
    return backup_path


def get_current_best_score() -> float:
    """Parse results.tsv to find the current best score."""
    if not RESULTS_FILE.exists():
        return 0.0
    lines = RESULTS_FILE.read_text().strip().split("\n")
    if len(lines) < 2:
        return 0.0
    best = 0.0
    for line in lines[1:]:
        parts = line.split("\t")
        if len(parts) >= 3:
            try:
                score = float(parts[2])
                # Ignore unrealistic scores (target < 24h artifacts per program.md)
                if score < 10:
                    best = max(best, score)
            except ValueError:
                continue
    return best


def get_next_experiment_nr() -> int:
    """Get next experiment number from results.tsv."""
    if not RESULTS_FILE.exists():
        return 1
    lines = RESULTS_FILE.read_text().strip().split("\n")
    if len(lines) < 2:
        return 1
    last_line = lines[-1]
    parts = last_line.split("\t")
    try:
        return int(parts[0]) + 1
    except (ValueError, IndexError):
        return len(lines)


def get_recent_results(n: int = MAX_RECENT_RESULTS) -> str:
    """Get last N lines from results.tsv (header + recent experiments)."""
    if not RESULTS_FILE.exists():
        return "(no results yet)"
    lines = RESULTS_FILE.read_text().strip().split("\n")
    header = lines[0]
    data_lines = lines[1:]
    recent = data_lines[-n:] if len(data_lines) > n else data_lines
    return header + "\n" + "\n".join(recent)


# ─── Search/Replace diff engine ─────────────────────────────────

def apply_search_replace(original: str, search: str, replace: str) -> tuple[str, bool]:
    """Apply a single search/replace operation.
    Returns (new_content, success).
    Tries exact match first, then fuzzy (stripped whitespace) match."""

    # Exact match
    if search in original:
        # Replace first occurrence only
        result = original.replace(search, replace, 1)
        return result, True

    # Fuzzy match: normalize whitespace and try again
    # This handles cases where the LLM slightly misremembers indentation
    search_lines = search.strip().splitlines()
    original_lines = original.splitlines()

    if len(search_lines) == 0:
        return original, False

    # Find the search block by matching stripped lines
    for start_idx in range(len(original_lines) - len(search_lines) + 1):
        match = True
        for j, search_line in enumerate(search_lines):
            if original_lines[start_idx + j].strip() != search_line.strip():
                match = False
                break
        if match:
            # Found it — preserve original indentation for the replacement
            # Detect indentation of the first matched line
            orig_first = original_lines[start_idx]
            orig_indent = len(orig_first) - len(orig_first.lstrip())
            replace_lines = replace.strip().splitlines()

            # Detect indentation of the replacement block
            if replace_lines:
                repl_first = replace_lines[0]
                repl_indent = len(repl_first) - len(repl_first.lstrip())
                indent_diff = orig_indent - repl_indent
            else:
                indent_diff = 0

            # Apply indentation adjustment
            adjusted_replace = []
            for rl in replace_lines:
                if indent_diff > 0:
                    adjusted_replace.append(" " * indent_diff + rl)
                elif indent_diff < 0:
                    # Remove excess indentation
                    strip_n = min(abs(indent_diff), len(rl) - len(rl.lstrip()))
                    adjusted_replace.append(rl[strip_n:])
                else:
                    adjusted_replace.append(rl)

            new_lines = (
                original_lines[:start_idx]
                + adjusted_replace
                + original_lines[start_idx + len(search_lines):]
            )
            return "\n".join(new_lines), True

    return original, False


def apply_all_diffs(original: str, diffs: list[dict]) -> tuple[str, int, int]:
    """Apply a list of search/replace diffs to original content.
    Returns (new_content, applied_count, failed_count)."""
    content = original
    applied = 0
    failed = 0
    for diff in diffs:
        search = diff.get("search", "")
        replace = diff.get("replace", "")
        content, success = apply_search_replace(content, search, replace)
        if success:
            applied += 1
        else:
            failed += 1
            print(f"    WARNING: Failed to apply diff (search block not found)")
            # Show first line of search for debugging
            first_line = search.strip().split("\n")[0][:80] if search.strip() else "(empty)"
            print(f"    Search starts with: {first_line}")
    return content, applied, failed


# ─── Prompt construction ────────────────────────────────────────

def build_system_prompt() -> str:
    return """You are an autonomous trading strategy optimizer. You modify strategy.py to maximize the `score` metric (higher = better).

RULES:
1. Modify ONLY strategy.py — prepare.py is read-only
2. No new dependencies (only torch, pandas, numpy are available)
3. Always change the OPIS variable to describe what you changed
4. target must be >= 24h (shorter targets create unrealistic scores)
5. DO NOT repeat experiments from the "NIE POWTARZAJ" section
6. Focus on ideas from the "Co warto probowac dalej" section

OUTPUT FORMAT — respond with EXACTLY this structure:

<thinking>
Your reasoning about what to try and why (1-3 paragraphs).
</thinking>

<new_opis>Short description of what you changed (this replaces the OPIS variable)</new_opis>

<diff>
<<<SEARCH
exact lines from strategy.py to find (include enough context for unique match)
===
replacement lines that will replace the search block
>>>
</diff>

You can include MULTIPLE search/replace blocks in a single <diff> section. Each block uses the <<<SEARCH ... === ... >>> format.

IMPORTANT:
- Copy the search lines EXACTLY from strategy.py (same indentation, same characters)
- Include 2-3 lines of unchanged context around your changes for reliable matching
- Keep changes focused — one idea at a time
- Always include one diff block that changes the OPIS variable"""


def build_user_prompt(strategy_code: str, results_text: str, program_text: str) -> str:
    return f"""## program.md (experiment protocol + record + rejected ideas)

{program_text}

## Recent results.tsv (last {MAX_RECENT_RESULTS} experiments)

{results_text}

## Current strategy.py (with line numbers for reference)

{_add_line_numbers(strategy_code)}

## Your task

Propose ONE focused improvement. Output your reasoning in <thinking> tags, the new OPIS in <new_opis> tags, and search/replace diffs in <diff> tags."""


def _add_line_numbers(code: str) -> str:
    """Add line numbers to code for LLM reference."""
    lines = code.split("\n")
    numbered = []
    for i, line in enumerate(lines, 1):
        numbered.append(f"{i:4d} | {line}")
    return "\n".join(numbered)


# ─── Response parsing ────────────────────────────────────────────

def extract_thinking(response: str) -> str:
    """Extract thinking/reasoning from LLM response."""
    match = re.search(r"<thinking>\s*\n?(.*?)\n?\s*</thinking>", response, re.DOTALL)
    if match:
        return match.group(1).strip()
    return response[:500].strip()


def extract_new_opis(response: str) -> str | None:
    """Extract new OPIS value from response."""
    match = re.search(r"<new_opis>\s*(.*?)\s*</new_opis>", response, re.DOTALL)
    if match:
        return match.group(1).strip()
    return None


def extract_diffs(response: str) -> list[dict]:
    """Extract search/replace blocks from response.
    Format:
    <<<SEARCH
    ... search text ...
    ===
    ... replace text ...
    >>>
    """
    diffs = []

    # Find the <diff> section
    diff_match = re.search(r"<diff>\s*\n?(.*?)\n?\s*</diff>", response, re.DOTALL)
    if not diff_match:
        return diffs

    diff_content = diff_match.group(1)

    # Parse individual blocks
    blocks = re.findall(
        r"<<<\s*SEARCH\s*\n(.*?)\n\s*===\s*\n(.*?)\n\s*>>>",
        diff_content,
        re.DOTALL,
    )

    for search, replace in blocks:
        # Strip line number prefixes if the LLM included them
        search_clean = _strip_line_numbers(search)
        replace_clean = _strip_line_numbers(replace)
        diffs.append({"search": search_clean, "replace": replace_clean})

    return diffs


def _strip_line_numbers(text: str) -> str:
    """Remove line number prefixes (e.g., '  42 | ') that the LLM may copy from the prompt."""
    lines = text.split("\n")
    stripped = []
    has_numbers = all(
        re.match(r"\s*\d+\s*\|\s?", line) or line.strip() == ""
        for line in lines if line.strip()
    )
    if has_numbers:
        for line in lines:
            match = re.match(r"\s*\d+\s*\|\s?(.*)", line)
            if match:
                stripped.append(match.group(1))
            else:
                stripped.append(line)
        return "\n".join(stripped)
    return text


def validate_modified_code(code: str) -> tuple[bool, str]:
    """Validate strategy.py after modifications."""
    checks = [
        ("def strategy(", "Missing strategy() function"),
        ("import pandas", "Missing pandas import"),
        ("import torch", "Missing torch import"),
        ("from prepare import", "Missing prepare import"),
        ("OPIS", "Missing OPIS variable"),
        ("evaluate(strategy", "Missing evaluate() call in __main__"),
    ]
    for pattern, msg in checks:
        if pattern not in code:
            return False, msg

    for pattern in _BLOCKED_PATTERNS:
        if pattern in code:
            return False, f"Blocked pattern detected: {pattern}"

    # Syntax check
    try:
        compile(code, "strategy.py", "exec")
    except SyntaxError as e:
        return False, f"Syntax error: {e}"

    return True, "OK"


# ─── Backtest runner ─────────────────────────────────────────────

def run_backtest(experiment_nr: int) -> tuple[float | None, str]:
    """Run python3 strategy.py and parse the score from output."""
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    log_path = LOGS_DIR / f"run_{experiment_nr}.log"

    # Kill any orphan strategy.py processes from previous failed runs
    try:
        subprocess.run(
            ["pkill", "-9", "-f", "python3 strategy.py"],
            capture_output=True, timeout=5
        )
        time.sleep(1)
    except Exception:
        pass

    try:
        result = subprocess.run(
            [sys.executable, str(STRATEGY_FILE)],
            capture_output=True,
            text=True,
            timeout=BACKTEST_TIMEOUT,
            cwd=str(PROJECT_DIR),
            env={
                **os.environ,
                "PYTHONUNBUFFERED": "1",
                "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True,max_split_size_mb:128",
            },
        )
        output = result.stdout + "\n" + result.stderr
        log_path.write_text(output, encoding="utf-8")

        if result.returncode != 0:
            return None, f"Backtest failed (exit code {result.returncode}):\n{result.stderr[-1000:]}"

        score_match = re.search(r"score:\s+([\d.]+)", output)
        if score_match:
            score = float(score_match.group(1))
            return score, output
        else:
            return None, f"Could not parse score from output:\n{output[-500:]}"

    except subprocess.TimeoutExpired:
        return None, f"Backtest timed out after {BACKTEST_TIMEOUT}s"
    except Exception as e:
        return None, f"Backtest error: {e}"


# ─── Experiment logging ──────────────────────────────────────────

def log_experiment(entry: dict):
    """Append experiment entry to JSONL log."""
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    with open(EXPERIMENT_LOG, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False, default=str) + "\n")


# ─── Main loop ───────────────────────────────────────────────────

def run_iteration(iteration: int, dry_run: bool = False) -> dict:
    """Run one autoresearch iteration. Returns experiment log entry."""
    print(f"\n{'='*70}")
    print(f"  AUTORESEARCH ITERATION {iteration}")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*70}\n")

    entry = {
        "iteration": iteration,
        "timestamp": datetime.now().isoformat(),
        "status": "unknown",
    }

    # Read current state
    current_score = get_current_best_score()
    entry["best_score_before"] = current_score
    print(f"  Current best score: {current_score:.4f}")

    strategy_code = read_file(STRATEGY_FILE)
    results_text = get_recent_results()
    program_text = read_file(PROGRAM_FILE) if PROGRAM_FILE.exists() else "(no program.md)"

    # Ensure Ollama is running (may have been stopped for VRAM in previous iteration)
    import urllib.request as _urlreq
    for _w in range(30):
        try:
            _rq = _urlreq.Request("http://localhost:11434/api/tags")
            with _urlreq.urlopen(_rq, timeout=5):
                break
        except Exception:
            # Try starting Ollama if not running
            if _w == 0:
                try:
                    subprocess.run(["systemctl", "start", "ollama"],
                                   capture_output=True, timeout=15)
                except Exception:
                    pass
            time.sleep(1)

    # Get LLM proposal
    print(f"  Querying Ollama ({MODEL})...")
    t0 = time.time()
    try:
        response = ollama_chat([
            {"role": "system", "content": build_system_prompt()},
            {"role": "user", "content": build_user_prompt(strategy_code, results_text, program_text)},
        ])
    except RuntimeError as e:
        entry["status"] = "llm_error"
        entry["error"] = str(e)
        print(f"  ERROR: {e}")
        log_experiment(entry)
        return entry

    llm_time = time.time() - t0
    entry["llm_time_sec"] = round(llm_time, 1)
    print(f"  LLM response received in {llm_time:.1f}s")

    # Save raw response for debugging
    debug_path = LOGS_DIR / f"autoresearch_raw_iter{iteration:04d}.txt"
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    debug_path.write_text(response, encoding="utf-8")

    # Extract thinking
    thinking = extract_thinking(response)
    entry["thinking"] = thinking
    print(f"\n  THINKING:\n  {thinking[:300]}{'...' if len(thinking) > 300 else ''}\n")

    # Extract new OPIS
    new_opis = extract_new_opis(response)
    if new_opis:
        entry["opis"] = new_opis
        print(f"  Proposed change: {new_opis}")

    # Extract diffs
    diffs = extract_diffs(response)
    if not diffs:
        entry["status"] = "parse_error"
        entry["error"] = "No search/replace diffs found in response"
        print(f"  ERROR: No diffs found in response")
        print(f"  Raw response saved to {debug_path}")
        log_experiment(entry)
        return entry

    entry["num_diffs"] = len(diffs)
    print(f"  Found {len(diffs)} diff block(s)")

    # Apply diffs to get modified code
    modified_code = strategy_code

    # If we got a new_opis, ensure the OPIS line is updated
    if new_opis:
        opis_match = re.search(r'(OPIS\s*=\s*["\'])(.+?)(["\'])', modified_code)
        if opis_match:
            old_opis_line = opis_match.group(0)
            new_opis_line = f'{opis_match.group(1)}{new_opis}{opis_match.group(3)}'
            modified_code = modified_code.replace(old_opis_line, new_opis_line, 1)

    modified_code, applied, failed = apply_all_diffs(modified_code, diffs)
    entry["diffs_applied"] = applied
    entry["diffs_failed"] = failed
    print(f"  Diffs: {applied} applied, {failed} failed")

    if applied == 0:
        entry["status"] = "diff_error"
        entry["error"] = "All diffs failed to apply"
        print(f"  ERROR: No diffs could be applied")
        log_experiment(entry)
        return entry

    # Validate modified code
    valid, msg = validate_modified_code(modified_code)
    if not valid:
        entry["status"] = "validation_error"
        entry["error"] = msg
        print(f"  ERROR: Validation failed — {msg}")
        log_experiment(entry)
        return entry

    if dry_run:
        entry["status"] = "dry_run"
        print(f"  DRY RUN — not applying change")
        dry_path = LOGS_DIR / f"autoresearch_proposed_iter{iteration:04d}.py"
        dry_path.write_text(modified_code, encoding="utf-8")
        print(f"  Proposed code saved to {dry_path}")
        log_experiment(entry)
        return entry

    # Backup current strategy.py
    backup_path = backup_strategy(iteration)
    entry["backup_path"] = str(backup_path)
    print(f"  Backup: {backup_path.name}")

    # Apply modified code
    write_file(STRATEGY_FILE, modified_code)
    print(f"  Applied changes to strategy.py")

    # Stop Ollama to free GPU VRAM for PyTorch training
    ollama_wait_for_vram(min_free_mib=20000, timeout_sec=30)

    # Run backtest
    experiment_nr = get_next_experiment_nr()
    entry["experiment_nr"] = experiment_nr
    print(f"  Running backtest (experiment #{experiment_nr})...")
    t0 = time.time()
    score, output = run_backtest(experiment_nr)
    backtest_time = time.time() - t0
    entry["backtest_time_sec"] = round(backtest_time, 1)

    # Restart Ollama for next iteration (stopped to free VRAM)
    try:
        subprocess.run(["systemctl", "start", "ollama"], capture_output=True, timeout=15)
    except Exception:
        pass

    if score is None:
        print(f"  BACKTEST FAILED ({backtest_time:.0f}s): {output[:200]}")
        entry["status"] = "backtest_error"
        entry["error"] = output[:500]
        shutil.copy2(backup_path, STRATEGY_FILE)
        print(f"  REVERTED to backup")
        log_experiment(entry)
        return entry

    entry["score"] = score
    print(f"  Score: {score:.4f} (backtest took {backtest_time:.0f}s)")

    # Compare with best
    if score > current_score:
        entry["status"] = "improvement"
        improvement_pct = ((score - current_score) / max(current_score, 0.001)) * 100
        entry["improvement_pct"] = round(improvement_pct, 2)
        print(f"\n  *** IMPROVEMENT: {current_score:.4f} -> {score:.4f} (+{improvement_pct:.1f}%) ***")
        print(f"  Keeping new strategy.py")
    else:
        entry["status"] = "no_improvement"
        print(f"\n  No improvement ({score:.4f} <= {current_score:.4f})")
        shutil.copy2(backup_path, STRATEGY_FILE)
        print(f"  REVERTED to backup")

    log_experiment(entry)
    return entry


def main():
    parser = argparse.ArgumentParser(description="Ollama-powered autoquant strategy optimizer")
    parser.add_argument("--max-iter", type=int, default=50,
                        help="Maximum iterations (default: 50)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Propose changes without applying")
    parser.add_argument("--temperature", type=float, default=0.7,
                        help="LLM temperature (default: 0.7)")
    parser.add_argument("--model", type=str, default=None,
                        help="Override Ollama model (default: from config)")
    args = parser.parse_args()

    if args.model:
        global MODEL
        MODEL = args.model

    print("=" * 70)
    print("  AUTOQUANT LOCAL AUTORESEARCH (Ollama)")
    print(f"  Model: {MODEL}")
    print(f"  Max iterations: {args.max_iter}")
    print(f"  Dry run: {args.dry_run}")
    print(f"  Project: {PROJECT_DIR}")
    print("=" * 70)

    # Verify Ollama is reachable
    try:
        import urllib.request
        with urllib.request.urlopen("http://localhost:11434/api/tags", timeout=5) as resp:
            tags = json.loads(resp.read())
            models = [m["name"] for m in tags.get("models", [])]
            if MODEL not in models:
                print(f"\n  WARNING: Model '{MODEL}' not found in Ollama.")
                print(f"  Available models: {', '.join(models)}")
                print(f"  Run: ollama pull {MODEL}")
                sys.exit(1)
            print(f"\n  Ollama OK — {MODEL} available")
    except Exception as e:
        print(f"\n  ERROR: Cannot reach Ollama: {e}")
        print(f"  Make sure Ollama is running: systemctl start ollama")
        sys.exit(1)

    if not STRATEGY_FILE.exists():
        print(f"\n  ERROR: {STRATEGY_FILE} not found")
        sys.exit(1)

    current_best = get_current_best_score()
    print(f"  Current best score: {current_best:.4f}")
    print()

    improvements = 0
    errors = 0
    total_time = time.time()
    iterations_done = 0

    for i in range(1, args.max_iter + 1):
        iterations_done = i
        try:
            entry = run_iteration(i, dry_run=args.dry_run)
            if entry["status"] == "improvement":
                improvements += 1
            elif entry["status"] in ("llm_error", "parse_error", "diff_error",
                                      "backtest_error", "validation_error"):
                errors += 1
        except KeyboardInterrupt:
            print(f"\n\n  Interrupted by user after {i-1} iterations")
            iterations_done = i - 1
            break
        except Exception as e:
            print(f"\n  UNEXPECTED ERROR in iteration {i}: {e}")
            errors += 1
            log_experiment({
                "iteration": i,
                "timestamp": datetime.now().isoformat(),
                "status": "crash",
                "error": str(e),
            })

    elapsed = time.time() - total_time
    final_score = get_current_best_score()

    print(f"\n\n{'='*70}")
    print(f"  AUTORESEARCH COMPLETE")
    print(f"  Total time: {elapsed/60:.1f} minutes")
    print(f"  Iterations: {iterations_done}")
    print(f"  Improvements: {improvements}")
    print(f"  Errors: {errors}")
    print(f"  Score: {current_best:.4f} -> {final_score:.4f}")
    print(f"  Log: {EXPERIMENT_LOG}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
