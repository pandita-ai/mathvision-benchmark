"""
generate.py — Generate diagram images from concise prompts using LLM APIs.

For code-generating LLMs: prompt → code response → detect format → compile → PNG
For image-generating models: prompt → image directly → PNG

Models: DeepSeek-V3, DeepSeek-R1, GPT-5.2 Thinking, o3-mini, Claude Opus 4.6,
Gemini 3.1 Pro, Qwen3.5-397B, Qwen3.5-35B, Llama 4 Maverick
"""

import argparse
import csv
import hashlib
import json
import os
import re
import subprocess
import sys
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from openai import OpenAI
from PIL import Image
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

MODELS = {
    # --- DeepSeek (DEEPSEEK_API_KEY) ---
    "deepseek-v3": {
        "type": "code_llm",
        "provider": "deepseek",
        "model_id": "deepseek-chat",
        "base_url": "https://api.deepseek.com",
        "env_key": "DEEPSEEK_API_KEY",
    },
    "deepseek-r1": {
        "type": "code_llm",
        "provider": "deepseek",
        "model_id": "deepseek-reasoner",
        "base_url": "https://api.deepseek.com",
        "env_key": "DEEPSEEK_API_KEY",
        "reasoning": True,
    },
    # --- OpenAI (OPENAI_API_KEY) ---
    "gpt-5.4": {
        "type": "code_llm",
        "provider": "openai",
        "model_id": "gpt-5.4",
        "base_url": "https://api.openai.com/v1",
        "env_key": "OPENAI_API_KEY",
        "reasoning": True,
    },
    "gpt-oss": {
        "type": "code_llm",
        "provider": "groq",
        "model_id": "openai/gpt-oss-120b",
        "base_url": "https://api.groq.com/openai/v1",
        "env_key": "GROQ_API_KEY",
    },
    # --- OpenRouter (OPENROUTER_API_KEY) ---
    "claude-opus-4.6": {
        "type": "code_llm",
        "provider": "openrouter",
        "model_id": "anthropic/claude-opus-4.6",
        "base_url": "https://openrouter.ai/api/v1",
        "env_key": "OPENROUTER_API_KEY",
    },
    "gemini-3.1-pro": {
        "type": "code_llm",
        "provider": "openrouter",
        "model_id": "google/gemini-3.1-pro-preview",
        "base_url": "https://openrouter.ai/api/v1",
        "env_key": "OPENROUTER_API_KEY",
    },
    "qwen3.5-397b": {
        "type": "code_llm",
        "provider": "openrouter",
        "model_id": "qwen/qwen3.5-397b-a17b",
        "base_url": "https://openrouter.ai/api/v1",
        "env_key": "OPENROUTER_API_KEY",
    },
    "qwen3.5-35b": {
        "type": "code_llm",
        "provider": "openrouter",
        "model_id": "qwen/qwen3.5-35b-a3b",
        "base_url": "https://openrouter.ai/api/v1",
        "env_key": "OPENROUTER_API_KEY",
        "max_tokens": 16384,
    },
    "llama-4-maverick": {
        "type": "code_llm",
        "provider": "openrouter",
        "model_id": "meta-llama/llama-4-maverick",
        "base_url": "https://openrouter.ai/api/v1",
        "env_key": "OPENROUTER_API_KEY",
    },
}

SYSTEM_PROMPT = """You are an expert at generating precise mathematical diagrams.
Given a description, produce code that renders the described diagram as an image.
Choose whichever format you believe will produce the best result: TikZ (LaTeX), SVG, Python (matplotlib), or any other approach.
Output ONLY the code inside a single code block. No explanation, no commentary."""

# ---------------------------------------------------------------------------
# Code extraction and format detection
# ---------------------------------------------------------------------------

def extract_code_block(response_text: str) -> tuple[str, str]:
    """Extract code from markdown code blocks and detect the format.

    Returns (code, format) where format is one of:
    'tikz', 'svg', 'python', 'unknown'
    """
    if not response_text:
        return "", "unknown"

    # Strip <think>...</think> blocks from reasoning models
    text = re.sub(r"<think>.*?</think>", "", response_text, flags=re.DOTALL).strip()
    if not text:
        text = response_text  # fallback if everything was in <think>

    # Find all fenced code blocks
    pattern = r"```(\w*)\n(.*?)```"
    matches = re.findall(pattern, text, re.DOTALL)

    if matches:
        # Use the LAST code block (reasoning models put the answer last)
        lang_hint, code = matches[-1]
        lang_hint = lang_hint.lower().strip()
    else:
        # No code block found — treat entire response as code
        code = text.strip()
        lang_hint = ""

    # Detect format
    fmt = detect_format(code, lang_hint)
    return code.strip(), fmt


def detect_format(code: str, lang_hint: str = "") -> str:
    """Auto-detect whether code is TikZ, SVG, Python/matplotlib, or unknown."""

    # Check language hint from code block
    if lang_hint in ("latex", "tex", "tikz"):
        return "tikz"
    if lang_hint in ("svg", "xml", "html"):
        return "svg"
    if lang_hint in ("python", "py", "python3"):
        return "python"

    # Content-based detection
    code_lower = code.lower()

    if "\\begin{tikzpicture}" in code or "\\tikz" in code_lower:
        return "tikz"
    if code.strip().startswith("<svg") or "<svg " in code_lower:
        return "svg"
    if "import matplotlib" in code or "plt.savefig" in code_lower or "plt.show" in code_lower:
        return "python"
    if "\\documentclass" in code or "\\begin{document}" in code:
        return "tikz"  # LaTeX document with likely TikZ
    if "import " in code and ("draw" in code_lower or "figure" in code_lower):
        return "python"

    return "unknown"


# ---------------------------------------------------------------------------
# Compilers: code → PNG image
# ---------------------------------------------------------------------------

def compile_tikz(code: str, output_path: str) -> bool:
    """Compile TikZ/LaTeX code to PNG."""
    # Wrap in document if not already wrapped
    if "\\documentclass" not in code:
        code = (
            "\\documentclass[border=2pt]{standalone}\n"
            "\\usepackage{tikz}\n"
            "\\usepackage{tikz-3dplot}\n"
            "\\usepackage{pgfplots}\n"
            "\\pgfplotsset{compat=1.18}\n"
            "\\usepackage{amsmath,amssymb,amsfonts}\n"
            "\\usepackage{xcolor}\n"
            "\\usepackage{graphicx}\n"
            "\\usepackage{lmodern}\n"
            "\\usepackage[T1]{fontenc}\n"
            "\\usepackage{circuitikz}\n"
            "\\usepackage{tikz-cd}\n"
            "\\usetikzlibrary{\n"
            "  calc,positioning,shapes.geometric,shapes.misc,shapes.symbols,\n"
            "  shapes.arrows,shapes.multipart,\n"
            "  decorations.pathmorphing,decorations.markings,decorations.pathreplacing,\n"
            "  decorations.text,\n"
            "  arrows.meta,arrows,\n"
            "  angles,quotes,\n"
            "  intersections,through,\n"
            "  patterns,shadings,\n"
            "  backgrounds,fit,\n"
            "  matrix,chains,trees,\n"
            "  3d,perspective,\n"
            "  plotmarks,\n"
            "  automata,petri,\n"
            "  mindmap,shadows,\n"
            "  spy,turtle,\n"
            "  folding,\n"
            "  babel,\n"
            "}\n"
            "\\begin{document}\n"
            f"{code}\n"
            "\\end{document}"
        )

    with tempfile.TemporaryDirectory() as tmpdir:
        tex_path = os.path.join(tmpdir, "diagram.tex")
        with open(tex_path, "w") as f:
            f.write(code)

        # LaTeX → PDF
        result = subprocess.run(
            ["pdflatex", "-interaction=nonstopmode", "-halt-on-error", "diagram.tex"],
            cwd=tmpdir,
            capture_output=True,
            timeout=30,
        )

        pdf_path = os.path.join(tmpdir, "diagram.pdf")
        if not os.path.exists(pdf_path):
            return False

        # PDF → PNG via pdftoppm
        result = subprocess.run(
            ["pdftoppm", "-png", "-r", "300", "-singlefile", pdf_path,
             os.path.join(tmpdir, "output")],
            capture_output=True,
            timeout=15,
        )

        png_path = os.path.join(tmpdir, "output.png")
        if os.path.exists(png_path):
            # Move to final destination
            Image.open(png_path).save(output_path)
            return True

    return False


def compile_svg(code: str, output_path: str) -> bool:
    """Compile SVG code to PNG."""
    import cairosvg

    # Ensure it's a complete SVG
    if not code.strip().startswith("<svg"):
        code = f'<svg xmlns="http://www.w3.org/2000/svg" width="800" height="600">\n{code}\n</svg>'

    try:
        cairosvg.svg2png(bytestring=code.encode("utf-8"), write_to=output_path,
                         output_width=800, output_height=800)
        return True
    except Exception:
        return False


def compile_python(code: str, output_path: str) -> bool:
    """Execute Python/matplotlib code and capture the saved figure."""

    with tempfile.TemporaryDirectory() as tmpdir:
        fig_path = os.path.join(tmpdir, "figure.png")

        # Replace plt.show() with savefig
        modified_code = code.replace("plt.show()", f"plt.savefig('{fig_path}', dpi=300, bbox_inches='tight')")

        # Redirect ALL savefig calls to our known path (handles hardcoded filenames)
        modified_code = re.sub(
            r"plt\.savefig\s*\([^)]*\)",
            f"plt.savefig('{fig_path}', dpi=300, bbox_inches='tight')",
            modified_code,
        )
        modified_code = re.sub(
            r"fig\.savefig\s*\([^)]*\)",
            f"fig.savefig('{fig_path}', dpi=300, bbox_inches='tight')",
            modified_code,
        )

        # If still no savefig, append one
        if "savefig" not in modified_code:
            modified_code += f"\nimport matplotlib.pyplot as plt\nplt.savefig('{fig_path}', dpi=300, bbox_inches='tight')\n"

        # Redirect PIL Image.save() calls to our path
        modified_code = re.sub(
            r"(\w+)\.save\s*\(\s*['\"][^'\"]+['\"]\s*\)",
            rf"\1.save('{fig_path}')",
            modified_code,
        )

        # Use non-interactive backend to avoid display issues
        modified_code = "import matplotlib\nmatplotlib.use('Agg')\n" + modified_code

        script_path = os.path.join(tmpdir, "render.py")
        with open(script_path, "w") as f:
            f.write(modified_code)

        result = subprocess.run(
            [sys.executable, script_path],
            capture_output=True,
            timeout=60,
            cwd=tmpdir,
        )

        # Check for any PNG in tmpdir as fallback
        if os.path.exists(fig_path):
            Image.open(fig_path).save(output_path)
            return True

    return False


def compile_code(code: str, fmt: str, output_path: str) -> bool:
    """Route to the appropriate compiler."""
    compilers = {
        "tikz": compile_tikz,
        "svg": compile_svg,
        "python": compile_python,
    }

    compiler = compilers.get(fmt)
    if compiler is None:
        # Try all compilers as fallback
        for name, comp in compilers.items():
            try:
                if comp(code, output_path):
                    return True
            except Exception:
                continue
        return False

    try:
        return compiler(code, output_path)
    except Exception:
        return False


# ---------------------------------------------------------------------------
# API client
# ---------------------------------------------------------------------------

def get_client(model_cfg: dict) -> OpenAI:
    """Create an OpenAI-compatible client for the given model config."""
    api_key = os.environ.get(model_cfg["env_key"])
    if not api_key:
        raise ValueError(f"Set {model_cfg['env_key']} environment variable")

    return OpenAI(
        api_key=api_key,
        base_url=model_cfg["base_url"],
    )


def generate_code(client: OpenAI, model_cfg: dict, prompt: str) -> str:
    """Send prompt to LLM and get code response."""
    params = {
        "model": model_cfg["model_id"],
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
    }

    if model_cfg.get("reasoning"):
        # Reasoning models: no temperature, use max_completion_tokens
        params["max_completion_tokens"] = model_cfg.get("max_tokens", 16384)
    else:
        params["temperature"] = 0
        params["max_tokens"] = model_cfg.get("max_tokens", 8192)

    # Extra provider-specific params (e.g., reasoning_effort for o3-mini)
    params.update(model_cfg.get("extra_params", {}))

    response = client.chat.completions.create(**params)
    msg = response.choices[0].message
    # Try content first, then reasoning_content (DeepSeek-R1), then reasoning (OpenRouter)
    result = msg.content or getattr(msg, "reasoning_content", None) or getattr(msg, "reasoning", None)
    return result or ""


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

# Thread-safe collections for parallel execution
_logs_lock = threading.Lock()
_stats_lock = threading.Lock()


def process_prompt(client, model_cfg, image_id, prompt, output_dir):
    """Process a single prompt: generate code → detect format → compile → save PNG.

    Returns (log_entry, status, format) — thread-safe, no shared mutable state.
    """
    img_path = os.path.join(output_dir, f"{image_id}.png")

    # Skip if already generated and valid
    if os.path.exists(img_path):
        try:
            img = Image.open(img_path)
            img.verify()
            return None, "cached", None
        except Exception:
            os.remove(img_path)  # corrupt PNG, regenerate

    # Step 1: Get code from LLM (with retry for transient errors)
    raw_response = None
    for attempt in range(3):
        try:
            raw_response = generate_code(client, model_cfg, prompt)
            break
        except Exception as e:
            if attempt < 2 and ("429" in str(e) or "500" in str(e) or "502" in str(e) or "503" in str(e) or "timeout" in str(e).lower()):
                time.sleep(2 ** attempt)
                continue
            return {"image_id": image_id, "error": str(e)}, "api_error", None

    if raw_response is None:
        return {"image_id": image_id, "error": "All retries failed"}, "api_error", None

    # Step 2: Extract code and detect format
    code, fmt = extract_code_block(raw_response)

    # Step 3: Build log entry
    log_entry = {
        "image_id": image_id,
        "format_detected": fmt,
        "code": code,
        "raw_response": raw_response,
    }

    # Step 4: Compile to image
    success = compile_code(code, fmt, img_path)
    log_entry["status"] = "success" if success else "compile_error"

    return log_entry, log_entry["status"], fmt


def main():
    parser = argparse.ArgumentParser(description="Generate diagram images from prompts")
    parser.add_argument("--model", default="deepseek-v3", choices=list(MODELS.keys()))
    parser.add_argument("--csv", default="data/concise_prompts.csv")
    parser.add_argument("--output-dir", default=None, help="Output directory (default: outputs/<model>)")
    parser.add_argument("--limit", type=int, default=None, help="Process only first N prompts")
    parser.add_argument("--offset", type=int, default=0, help="Start from prompt N")
    parser.add_argument("--workers", type=int, default=1, help="Number of parallel workers (default: 1)")
    args = parser.parse_args()

    model_cfg = MODELS[args.model]
    output_dir = args.output_dir or os.path.join("outputs", args.model)
    os.makedirs(output_dir, exist_ok=True)

    # Load prompts
    with open(args.csv, "r") as f:
        reader = csv.DictReader(f)
        prompts = list(reader)

    if args.offset:
        prompts = prompts[args.offset:]
    if args.limit:
        prompts = prompts[:args.limit]

    print(f"Model: {args.model} ({model_cfg['model_id']})")
    print(f"Prompts: {len(prompts)}")
    print(f"Workers: {args.workers}")
    print(f"Output: {output_dir}")

    # Init API client (thread-safe — OpenAI client uses httpx connection pooling)
    client = get_client(model_cfg)

    logs = []
    stats = {"success": 0, "compile_error": 0, "api_error": 0, "cached": 0}
    format_counts = {}

    if args.workers <= 1:
        # Sequential mode (original behavior)
        for row in tqdm(prompts, desc=f"Generating ({args.model})"):
            log_entry, status, fmt = process_prompt(
                client, model_cfg,
                row["image_id"], row["concise_prompt"],
                output_dir
            )
            stats[status] = stats.get(status, 0) + 1
            if log_entry:
                logs.append(log_entry)
            if fmt:
                format_counts[fmt] = format_counts.get(fmt, 0) + 1
            time.sleep(0.5)
    else:
        # Parallel mode
        pbar = tqdm(total=len(prompts), desc=f"Generating ({args.model}, {args.workers}w)")

        def _worker(row):
            log_entry, status, fmt = process_prompt(
                client, model_cfg,
                row["image_id"], row["concise_prompt"],
                output_dir
            )
            return log_entry, status, fmt

        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = {executor.submit(_worker, row): row for row in prompts}
            for future in as_completed(futures):
                try:
                    log_entry, status, fmt = future.result()
                except Exception as e:
                    status = "api_error"
                    log_entry, fmt = None, None

                stats[status] = stats.get(status, 0) + 1
                if log_entry:
                    logs.append(log_entry)
                if fmt:
                    format_counts[fmt] = format_counts.get(fmt, 0) + 1
                pbar.update(1)

        pbar.close()

    # Save logs
    log_path = os.path.join(output_dir, "generation_log.json")
    with open(log_path, "w") as f:
        json.dump({"stats": stats, "format_counts": format_counts, "entries": logs}, f, indent=2)

    print(f"\n--- Results ---")
    print(f"Success:       {stats['success']}")
    print(f"Compile error: {stats['compile_error']}")
    print(f"API error:     {stats['api_error']}")
    print(f"Cached:        {stats['cached']}")
    print(f"Format distribution: {format_counts}")
    print(f"Log saved to: {log_path}")


if __name__ == "__main__":
    main()
