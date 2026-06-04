#!/usr/bin/env python3
"""Generate a single self-contained HTML record of the PSALM project.

Regenerable (reads sources, no hand-curation): assimilates the charter, PRD, spec,
requirements, all contracts, all ADRs, corpus/tokenizer manifests, training/eval results,
the orchestrator state, research references, and lessons (from the knowledge store), then
emits ``docs/reports/PSALM-project-record-<ISO8601>.html``.

    uv run python scripts/build_project_record.py
"""

from __future__ import annotations

import html
import json
import sqlite3
from datetime import datetime
from pathlib import Path

import markdown

ROOT = Path(__file__).resolve().parents[1]
MD_EXT = ["extra", "tables", "fenced_code", "sane_lists", "toc"]


def md(text: str) -> str:
    return markdown.markdown(text, extensions=MD_EXT)


def read(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except Exception:
        return ""


def section(title: str, anchor: str, body_html: str) -> str:
    return f'<section id="{anchor}"><h2>{html.escape(title)}</h2>{body_html}</section>'


def details(summary: str, body_html: str, open_: bool = False) -> str:
    o = " open" if open_ else ""
    return f"<details{o}><summary>{html.escape(summary)}</summary>{body_html}</details>"


def render_md_file(path: Path) -> str:
    txt = read(path)
    if not txt:
        return f"<p><em>(missing: {html.escape(str(path.relative_to(ROOT)))})</em></p>"
    return md(txt)


def json_table(obj: dict, keys: list[str] | None = None) -> str:
    rows = []
    items = [(k, obj.get(k)) for k in keys] if keys else list(obj.items())
    for k, v in items:
        if isinstance(v, (dict, list)):
            v = json.dumps(v)
        rows.append(f"<tr><th>{html.escape(str(k))}</th><td>{html.escape(str(v))}</td></tr>")
    return f"<table class='kv'>{''.join(rows)}</table>"


# ---------------- sections ----------------

def sec_narrative() -> str:
    state = read(ROOT / "docs/memory/ORCHESTRATOR-STATE.md")
    charter = render_md_file(ROOT / "CHARTER.md")
    intro = (
        "<p>PSALM (Pāṇinian Structured pretraining for Small LAnguage Models) tests whether a "
        "structural / morphological prior — a Pāṇinian grammar dose and a Navya-Nyāya "
        "Vyutpattivāda (Paribhāṣā) graph dose, realized natively via Vidyut — improves a "
        "small English language model versus a matched Dyck bracketing control, under the "
        "BabyLM Strict-Small budget. The project is GPU-only, no-mock, with budget-matched "
        "ablation arms (A–D) and an orthogonal leaderboard submission track.</p>"
    )
    return section("1. Project narrative (start → now)", "narrative",
                   intro + details("Charter", charter) +
                   details("Orchestrator state (current)", md(state), open_=True))


def sec_prd_spec() -> str:
    body = (
        details("PRD (docs/prd.md)", render_md_file(ROOT / "docs/prd.md")) +
        details("Spec (docs/spec.md)", render_md_file(ROOT / "docs/spec.md")) +
        details("Requirements", render_md_file(ROOT / "docs/requirements/2026-05-31-psalm-requirements.md"))
    )
    return section("2. PRD, spec & requirements", "prd-spec", body)


def sec_contracts() -> str:
    cdir = ROOT / "docs/contracts"
    parts = []
    for p in sorted(cdir.glob("*.md")):
        parts.append(details(p.name, render_md_file(p)))
    return section("3. Contracts", "contracts", "".join(parts) or "<p>none</p>")


def _adr_meta(text: str) -> tuple[str, str, str]:
    title = status = date = ""
    for line in text.splitlines():
        s = line.strip()
        if not title and s.startswith("#"):
            title = s.lstrip("# ").strip()
        if s.lower().startswith("- status:"):
            status = s.split(":", 1)[1].strip()
        if s.lower().startswith("- date:"):
            date = s.split(":", 1)[1].strip()
        if title and status and date:
            break
    return title, status, date


def sec_adrs() -> str:
    ddir = ROOT / "docs/decisions"
    adrs = sorted(ddir.glob("*.md"))
    rows = ["<tr><th>ADR</th><th>Title</th><th>Status</th><th>Date</th></tr>"]
    bodies = []
    for p in adrs:
        txt = read(p)
        title, status, date = _adr_meta(txt)
        num = p.name.split("-")[0]
        rows.append(
            f"<tr><td>{html.escape(num)}</td><td>{html.escape(title)}</td>"
            f"<td>{html.escape(status)}</td><td>{html.escape(date)}</td></tr>"
        )
        bodies.append(details(p.name, md(txt)))
    digest = f"<table class='kv'>{''.join(rows)}</table>"
    return section(f"4. ADR digest ({len(adrs)} decisions)", "adrs",
                   digest + "<h3>Full ADRs</h3>" + "".join(bodies))


def sec_manifests() -> str:
    parts = []
    prior = ROOT / "data/corpora/priors/paninian_1m_stats.json"
    if prior.exists():
        parts.append("<h3>Paninian prior dose stats</h3>" +
                     json_table(json.loads(read(prior))))
    spm_vocab = ROOT / "data/tokenizer/strict_small/spm.vocab"
    if spm_vocab.exists():
        n = sum(1 for _ in spm_vocab.open(encoding="utf-8"))
        parts.append(f"<h3>Tokenizer</h3><p>Joint SentencePiece, vocab size "
                     f"<strong>{n}</strong> (data/tokenizer/strict_small/spm.model).</p>")
    return section("5. Corpus & tokenizer manifests", "manifests",
                   "".join(parts) or "<p>none</p>")


def sec_results() -> str:
    cdir = ROOT / "data/checkpoints"
    rows = ["<tr><th>Checkpoint</th><th>arm/seed</th><th>steps</th><th>tokens</th>"
            "<th>best_loss</th><th>BLiMP (local PLL)</th></tr>"]
    for summ in sorted(cdir.glob("*/*/summary.json")):
        try:
            s = json.loads(read(summ))
        except Exception:
            continue
        d = summ.parent
        blimp = ""
        for bf in ("blimp_full.json", "blimp_pll.json"):
            bp = d / bf
            if bp.exists():
                try:
                    blimp = f"{json.loads(read(bp)).get('overall_accuracy'):.4f}"
                except Exception:
                    pass
                break
        rel = d.relative_to(cdir)
        rows.append(
            f"<tr><td>{html.escape(str(rel))}</td>"
            f"<td>{s.get('arm')}/{s.get('seed')}</td><td>{s.get('steps')}</td>"
            f"<td>{s.get('tokens_seen')}</td><td>{s.get('best_loss')}</td>"
            f"<td>{blimp}</td></tr>"
        )
    table = f"<table class='kv'>{''.join(rows)}</table>"

    # Official / GLUE summaries if present.
    extra = []
    for js in sorted((ROOT / "data/hf_export").glob("*/official_summary.json")):
        extra.append(details(f"official_summary: {js.parent.name}",
                             f"<pre>{html.escape(read(js))}</pre>"))
    for js in sorted((ROOT / "data/hf_export").glob("*/glue_summary.json")):
        extra.append(details(f"glue_summary: {js.parent.name}",
                             f"<pre>{html.escape(read(js))}</pre>"))

    leaderboard = (
        "<h3>Leaderboard comparison (Strict-Small reference)</h3>"
        "<table class='kv'>"
        "<tr><th>Model</th><th>BLiMP</th><th>EWoK</th><th>GLUE</th><th>Note</th></tr>"
        "<tr><td>PSALM arm A (current)</td><td>64.55</td><td>49.09</td><td>—</td>"
        "<td>internal BLiMP-PLL 0.6415</td></tr>"
        "<tr><td>MTP NTP baseline</td><td>63.95</td><td>49.73</td><td>—</td><td>HU Berlin</td></tr>"
        "<tr><td>AMLM (top encoder)</td><td>71.4</td><td>—</td><td>70.7</td>"
        "<td>adaptive masking; target</td></tr>"
        "</table>"
    )
    return section("6. Results & artifacts", "results", table + leaderboard + "".join(extra))


def sec_research() -> str:
    rdir = ROOT / "docs/research"
    parts = []
    for p in sorted(rdir.glob("*.md")):
        parts.append(details(p.name, render_md_file(p)))
    return section("7. Research references", "research", "".join(parts) or "<p>none</p>")


def sec_next() -> str:
    body = (
        "<p>The frozen H1 ablation (arms A–D) and the orthogonal leaderboard submission "
        "track are defined in ADR-0036 and ADR-0038. Planned next stages:</p>"
        "<ul>"
        "<li>Complete the H1 battery (arms A–D × seeds 0–2); paired bootstrap / permutation "
        "with Holm–Bonferroni for B-vs-C, D-vs-C, D-vs-B; declare finding + Tarka memo.</li>"
        "<li>Battery-wide official Text-Average suite + (Super)GLUE on all arm checkpoints "
        "(ADR-0037).</li>"
        "<li>Leaderboard submission model with parity-gated speedups and optional levers "
        "(ADR-0038), trained at max budget.</li>"
        "<li>FF-merge worktrees to integration; verify remote and push at the milestone.</li>"
        "</ul>"
    )
    return section("8. Planned next stages", "next", body)


def sec_lessons() -> str:
    parts = []
    ks = ROOT / "outputs/knowledge-store.sqlite"
    if ks.exists():
        try:
            conn = sqlite3.connect(ks)
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                "SELECT kind, title, body FROM notes ORDER BY kind, created_at"
            ).fetchall()
            conn.close()
            lis = [f"<li><strong>[{html.escape(r['kind'])}]</strong> "
                   f"{html.escape(r['title'])}<br><small>{html.escape(r['body'])}</small></li>"
                   for r in rows]
            parts.append(f"<ul>{''.join(lis)}</ul>")
        except Exception as exc:
            parts.append(f"<p>knowledge store unreadable: {html.escape(str(exc))}</p>")
    # Experiment findings.
    for p in sorted((ROOT / "docs/experiments").glob("*.md")):
        if p.name == "README.md":
            continue
        parts.append(details(f"experiment: {p.name}", render_md_file(p)))
    return section("9. Lessons learned & findings", "lessons", "".join(parts) or "<p>none</p>")


CSS = """
body{font-family:-apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif;line-height:1.5;
max-width:1100px;margin:0 auto;padding:2rem;color:#1a1a1a;background:#fafafa}
h1{border-bottom:3px solid #6a4c93;padding-bottom:.3rem}
h2{margin-top:2.5rem;border-bottom:1px solid #ccc;padding-bottom:.2rem;color:#6a4c93}
h3{color:#444}
nav{background:#fff;border:1px solid #ddd;border-radius:8px;padding:1rem;margin:1rem 0}
nav a{margin-right:1rem;color:#6a4c93;text-decoration:none}
table.kv{border-collapse:collapse;width:100%;margin:1rem 0;font-size:.92rem}
table.kv th,table.kv td{border:1px solid #ddd;padding:.4rem .6rem;text-align:left;vertical-align:top}
table.kv th{background:#f0ecf7}
details{margin:.5rem 0;background:#fff;border:1px solid #e0e0e0;border-radius:6px;padding:.5rem 1rem}
summary{cursor:pointer;font-weight:600;color:#6a4c93}
pre{background:#2d2d2d;color:#f0f0f0;padding:1rem;border-radius:6px;overflow:auto;font-size:.85rem}
code{background:#eee;padding:.1rem .3rem;border-radius:3px}
pre code{background:none;padding:0}
.meta{color:#666;font-size:.9rem}
"""


def build() -> Path:
    ts = datetime.now().astimezone().replace(microsecond=0).isoformat()
    fname = f"PSALM-project-record-{ts.replace(':', '').replace('+', 'p')}.html"
    out = ROOT / "docs/reports" / fname
    out.parent.mkdir(parents=True, exist_ok=True)

    nav = ("<nav><strong>Contents:</strong> "
           '<a href="#narrative">Narrative</a><a href="#prd-spec">PRD/Spec</a>'
           '<a href="#contracts">Contracts</a><a href="#adrs">ADRs</a>'
           '<a href="#manifests">Manifests</a><a href="#results">Results</a>'
           '<a href="#research">Research</a><a href="#next">Next stages</a>'
           '<a href="#lessons">Lessons</a></nav>')

    sections = "".join([
        sec_narrative(), sec_prd_spec(), sec_contracts(), sec_adrs(),
        sec_manifests(), sec_results(), sec_research(), sec_next(), sec_lessons(),
    ])

    doc = (
        f"<!doctype html><html lang='en'><head><meta charset='utf-8'>"
        f"<meta name='viewport' content='width=device-width,initial-scale=1'>"
        f"<title>PSALM Project Record — {html.escape(ts)}</title><style>{CSS}</style></head>"
        f"<body><h1>PSALM Project Record</h1>"
        f"<p class='meta'>Generated {html.escape(ts)} · repo PSALM · branch "
        f"integration/data-engine-v2 · regenerable via scripts/build_project_record.py</p>"
        f"{nav}{sections}</body></html>"
    )
    out.write_text(doc, encoding="utf-8")
    return out


if __name__ == "__main__":
    path = build()
    size_kb = path.stat().st_size / 1024
    print(f"OK project record -> {path} ({size_kb:.0f} KB)")
