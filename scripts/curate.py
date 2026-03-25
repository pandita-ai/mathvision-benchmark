#!/usr/bin/env python3
"""
curate.py — Local curation UI for MathVision dataset.
Opens a browser with ground truth images + prompts. Check boxes to exclude rows.
Exports excluded_ids.txt on save.

Usage:
    python3 scripts/curate.py
    python3 scripts/curate.py --category "combinatorics"   # filter by category
    python3 scripts/curate.py --port 8765
"""

import argparse
import csv
import base64
import json
import os
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

ROOT = Path(__file__).parent.parent
CSV_PATH = ROOT / "data" / "concise_prompts.csv"
GT_DIR = ROOT / "data" / "ground_truth"
EXCLUSIONS_PATH = ROOT / "data" / "excluded_ids.txt"


def load_rows(category_filter=None):
    rows = []
    with open(CSV_PATH) as f:
        for row in csv.DictReader(f):
            if category_filter and category_filter.lower() not in row["category"].lower():
                continue
            rows.append(row)
    return rows


def load_exclusions():
    if EXCLUSIONS_PATH.exists():
        return set(EXCLUSIONS_PATH.read_text().splitlines())
    return set()


def image_to_data_uri(image_id):
    path = GT_DIR / f"{image_id}.png"
    if not path.exists():
        return None
    data = base64.b64encode(path.read_bytes()).decode()
    return f"data:image/png;base64,{data}"


def build_html(rows, exclusions, page, per_page, category_filter):
    total = len(rows)
    total_pages = max(1, (total + per_page - 1) // per_page)
    page = max(1, min(page, total_pages))
    start = (page - 1) * per_page
    page_rows = rows[start:start + per_page]

    cards = []
    for row in page_rows:
        iid = row["image_id"]
        checked = "checked" if iid in exclusions else ""
        img_uri = image_to_data_uri(iid)
        img_tag = f'<img src="{img_uri}" alt="GT {iid}">' if img_uri else '<div class="no-img">No image</div>'
        prompt = row["concise_prompt"].replace("<", "&lt;").replace(">", "&gt;")
        category = row["category"].replace("<", "&lt;")
        cards.append(f"""
        <div class="card {'excluded' if checked else ''}" id="card-{iid}">
            <div class="card-header">
                <label class="checkbox-wrap">
                    <input type="checkbox" class="exclude-cb" data-id="{iid}" {checked} onchange="toggleCard(this)">
                    <span class="cb-label">Exclude</span>
                </label>
                <span class="meta">ID: <strong>{iid}</strong> &nbsp;|&nbsp; {category}</span>
            </div>
            <div class="card-body">
                <div class="img-wrap">{img_tag}</div>
                <div class="prompt">{prompt}</div>
            </div>
        </div>""")

    cards_html = "\n".join(cards)
    cat_param = f"&category={category_filter}" if category_filter else ""
    excluded_count = len(exclusions)

    pagination = ""
    if total_pages > 1:
        prev_disabled = "disabled" if page == 1 else ""
        next_disabled = "disabled" if page == total_pages else ""
        pagination = f"""
        <div class="pagination">
            <button onclick="navigate('/?page={page-1}&per_page={per_page}{cat_param}')" class="btn {prev_disabled}">&#8592; Prev</button>
            <span>Page {page} of {total_pages} &nbsp;({total} rows)</span>
            <button onclick="navigate('/?page={page+1}&per_page={per_page}{cat_param}')" class="btn {next_disabled}">Next &#8594;</button>
        </div>"""

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>MathVision Dataset Curation</title>
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; background: #f0f2f5; color: #1a1a2e; }}
  .topbar {{ background: #1a1a2e; color: white; padding: 14px 24px; display: flex; align-items: center; justify-content: space-between; position: sticky; top: 0; z-index: 100; box-shadow: 0 2px 8px rgba(0,0,0,0.3); }}
  .topbar h1 {{ font-size: 1.1rem; font-weight: 600; }}
  .topbar .stats {{ font-size: 0.85rem; opacity: 0.8; }}
  .actions {{ display: flex; gap: 10px; align-items: center; }}
  .btn {{ background: #4f8ef7; color: white; border: none; padding: 8px 16px; border-radius: 6px; cursor: pointer; font-size: 0.85rem; text-decoration: none; display: inline-block; transition: background 0.2s; }}
  .btn:hover {{ background: #3a7be0; }}
  .btn.danger {{ background: #e05c5c; }}
  .btn.danger:hover {{ background: #c94444; }}
  .btn.success {{ background: #27ae60; }}
  .btn.success:hover {{ background: #1e8c4a; }}
  .btn.disabled {{ pointer-events: none; opacity: 0.4; }}
  .container {{ max-width: 1400px; margin: 0 auto; padding: 20px; }}
  .toolbar {{ display: flex; gap: 12px; margin-bottom: 16px; align-items: center; flex-wrap: wrap; }}
  .toolbar input {{ padding: 8px 12px; border: 1px solid #ccc; border-radius: 6px; font-size: 0.9rem; width: 220px; }}
  .toolbar select {{ padding: 8px 12px; border: 1px solid #ccc; border-radius: 6px; font-size: 0.9rem; }}
  .grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(360px, 1fr)); gap: 16px; }}
  .card {{ background: white; border-radius: 10px; overflow: hidden; box-shadow: 0 1px 4px rgba(0,0,0,0.1); border: 2px solid transparent; transition: border-color 0.2s, box-shadow 0.2s; }}
  .card.excluded {{ border-color: #e05c5c; background: #fff8f8; }}
  .card-header {{ padding: 10px 14px; background: #f7f8fa; border-bottom: 1px solid #eee; display: flex; align-items: center; justify-content: space-between; }}
  .card.excluded .card-header {{ background: #fdecea; }}
  .checkbox-wrap {{ display: flex; align-items: center; gap: 8px; cursor: pointer; }}
  .exclude-cb {{ width: 16px; height: 16px; cursor: pointer; accent-color: #e05c5c; }}
  .cb-label {{ font-size: 0.82rem; font-weight: 600; color: #e05c5c; }}
  .meta {{ font-size: 0.78rem; color: #666; }}
  .card-body {{ display: flex; gap: 12px; padding: 12px 14px; }}
  .img-wrap {{ flex: 0 0 160px; }}
  .img-wrap img {{ width: 160px; height: 120px; object-fit: contain; border-radius: 4px; border: 1px solid #eee; background: #fafafa; }}
  .no-img {{ width: 160px; height: 120px; display: flex; align-items: center; justify-content: center; background: #f0f0f0; color: #aaa; font-size: 0.8rem; border-radius: 4px; }}
  .prompt {{ font-size: 0.78rem; line-height: 1.5; color: #444; overflow-y: auto; max-height: 120px; flex: 1; }}
  .pagination {{ display: flex; justify-content: center; align-items: center; gap: 16px; margin: 24px 0; font-size: 0.9rem; }}
  .save-banner {{ position: fixed; bottom: 20px; right: 20px; background: #1a1a2e; color: white; padding: 12px 20px; border-radius: 8px; font-size: 0.9rem; display: none; box-shadow: 0 4px 12px rgba(0,0,0,0.3); }}
  .save-banner.show {{ display: block; }}
</style>
</head>
<body>
<div class="topbar">
  <h1>MathVision Curation &nbsp;<span style="opacity:0.5;font-weight:400">({total} rows shown)</span></h1>
  <div class="stats">Excluded: <strong id="excl-count">{excluded_count}</strong> / 3040</div>
  <div class="actions">
    <span id="save-status" style="font-size:0.8rem;opacity:0.7"></span>
    <button class="btn danger" onclick="selectPage()">Exclude Page</button>
    <button class="btn" onclick="clearPage()">Clear Page</button>
  </div>
</div>

<div class="container">
  <div class="toolbar">
    <input type="text" id="search" placeholder="Search prompt..." oninput="filterCards()">
    <select onchange="navigate('/?page=1&per_page='+this.value+'{cat_param}')">
      {''.join(f'<option value="{n}" {"selected" if n==per_page else ""}>{n} per page</option>' for n in [12, 24, 48, 96])}
    </select>
  </div>

  <div class="grid" id="grid">
    {cards_html}
  </div>

  {pagination}
</div>

<script>
const excluded = new Set({json.dumps(list(exclusions))});
let saveTimer = null;

function persistNow() {{
  return fetch('/save', {{
    method: 'POST',
    headers: {{'Content-Type': 'application/json'}},
    body: JSON.stringify({{excluded: Array.from(excluded)}})
  }}).then(r => r.json()).then(d => {{
    document.getElementById('save-status').textContent = `Saved (${{d.count}} excluded)`;
  }});
}}

function scheduleAutoSave() {{
  clearTimeout(saveTimer);
  document.getElementById('save-status').textContent = 'Saving...';
  saveTimer = setTimeout(persistNow, 400);
}}

function toggleCard(cb) {{
  const id = cb.dataset.id;
  const card = document.getElementById('card-' + id);
  if (cb.checked) {{
    excluded.add(id);
    card.classList.add('excluded');
  }} else {{
    excluded.delete(id);
    card.classList.remove('excluded');
  }}
  document.getElementById('excl-count').textContent = excluded.size;
  scheduleAutoSave();
}}

function selectPage() {{
  document.querySelectorAll('.exclude-cb').forEach(cb => {{
    if (!cb.checked) {{ cb.checked = true; toggleCard(cb); }}
  }});
}}

function clearPage() {{
  document.querySelectorAll('.exclude-cb').forEach(cb => {{
    if (cb.checked) {{ cb.checked = false; toggleCard(cb); }}
  }});
}}

function filterCards() {{
  const q = document.getElementById('search').value.toLowerCase();
  document.querySelectorAll('.card').forEach(card => {{
    const text = card.querySelector('.prompt').textContent.toLowerCase();
    card.style.display = text.includes(q) ? '' : 'none';
  }});
}}

function navigate(url) {{
  persistNow().then(() => {{ location.href = url; }});
}}
</script>
</body>
</html>"""


class Handler(BaseHTTPRequestHandler):
    rows = []
    exclusions = set()
    category_filter = None

    def log_message(self, format, *args):
        pass  # suppress request logs

    def do_GET(self):
        parsed = urlparse(self.path)
        params = parse_qs(parsed.query)
        page = int(params.get("page", [1])[0])
        per_page = int(params.get("per_page", [24])[0])
        html = build_html(self.rows, self.exclusions, page, per_page, self.category_filter)
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.end_headers()
        self.wfile.write(html.encode())

    def do_POST(self):
        if self.path == "/save":
            length = int(self.headers["Content-Length"])
            body = json.loads(self.rfile.read(length))
            ids = sorted(body["excluded"], key=lambda x: int(x) if x.isdigit() else x)
            EXCLUSIONS_PATH.write_text("\n".join(ids) + ("\n" if ids else ""))
            Handler.exclusions = set(ids)
            resp = json.dumps({"count": len(ids)}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(resp)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8766)
    parser.add_argument("--category", type=str, default=None)
    parser.add_argument("--per-page", type=int, default=24)
    args = parser.parse_args()

    Handler.rows = load_rows(args.category)
    Handler.exclusions = load_exclusions()
    Handler.category_filter = args.category

    print(f"Loaded {len(Handler.rows)} rows, {len(Handler.exclusions)} existing exclusions")
    print(f"Opening http://localhost:{args.port}")
    import webbrowser
    webbrowser.open(f"http://localhost:{args.port}")

    server = HTTPServer(("localhost", args.port), Handler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped. Exclusions saved to data/excluded_ids.txt")


if __name__ == "__main__":
    main()
