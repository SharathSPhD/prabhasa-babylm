#!/usr/bin/env python3
"""Fetch the BabyLM 2026 leaderboard tables via the Space API (read-only; no new
submission is created because we pass a None results-file, which makes the
handler return early before writing anything) and compute our rank per track.
"""
from gradio_client import Client

SPACE = "BabyLM-community/BabyLM-Leaderboard-2026"
OURS = {"strict": "prabhasa-b_s", "strict-small": "prabhasa-b_ss-0.1"}

c = Client(SPACE)
# 39 args; param_4 (results_file)=None -> add_new_eval returns early, no write.
res = c.predict(
    "x", "main", "x", "strict", None, None,        # 0-5
    "Decoder only", [], "GPT-2",                    # 6-8 dropdowns (valid defaults)
    "cosine", 0, "BPE", "0", 0, 0, 0,              # 9-15
    "BabyLM strict", 0, "", 0.0, "AdamW", 0, 0, 0, # 16-23
    0, 0, 0, "Not applicable", "", "Not applicable",  # 24-29
    "", None, "", False, False, False, 0, 0, 0,    # 30-38
    api_name="/submit_and_refresh",
)
# res = (message, lb_strict, lb_strict_small, lb_multilingual)
tracks = {"strict": res[1], "strict-small": res[2]}

def colidx(headers, *names):
    for n in names:
        for i, h in enumerate(headers):
            if n.lower() in str(h).lower():
                return i
    return None

for track, lb in tracks.items():
    headers = lb["headers"]; rows = lb["data"]
    mi = colidx(headers, "Model", "eval_name")
    ti = colidx(headers, "Text Average", "Average")
    bi = colidx(headers, "BLiMP")
    gi = colidx(headers, "GLUE")
    print(f"\n===== {track.upper()}  ({len(rows)} entries) =====")
    print("columns:", headers)
    # locate our row
    me = None
    for r in rows:
        if OURS[track] in str(r[mi]):
            me = r; break
    if me is None:
        print("  OUR ROW NOT FOUND"); continue
    def rank_by(idx, label):
        def val(r):
            try: return float(r[idx])
            except: return float("-inf")
        order = sorted(rows, key=val, reverse=True)
        pos = next(i for i, r in enumerate(order, 1) if r is me)
        print(f"  rank by {label:14s}: #{pos} / {len(rows)}   (our value = {me[idx]})")
    if ti is not None: rank_by(ti, "Text Average")
    if bi is not None: rank_by(bi, "BLiMP")
    if gi is not None: rank_by(gi, "(Super)GLUE")
    # show the display order position too (as the table renders it)
    disp_pos = next((i for i, r in enumerate(rows, 1) if r is me), None)
    print(f"  position in default table order: #{disp_pos} / {len(rows)}")
