"""Generate prediction_discrepancy_root_cause.docx from Snowflake CSV data."""
from __future__ import annotations
import csv, json
from datetime import datetime, timezone
from pathlib import Path

from docx import Document
from docx.shared import Pt, Inches, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.oxml.ns import qn
from docx.oxml import OxmlElement

NOW = datetime(2026, 10, 25, 18, 0, 0, tzinfo=timezone.utc)
CSV_PATH = Path("C:/Users/CL4QR/Downloads/webatds_subservice_column_schema_2026-10-25-1818.csv")
OUT_PATH = Path("C:/Users/CL4QR/Downloads/prediction_discrepancy_root_cause.docx")


# ── helpers ───────────────────────────────────────────────────────────────────

def shade_row(row, hex_color: str):
    for cell in row.cells:
        tc = cell._tc
        tcPr = tc.get_or_add_tcPr()
        shd = OxmlElement("w:shd")
        shd.set(qn("w:val"), "clear")
        shd.set(qn("w:color"), "auto")
        shd.set(qn("w:fill"), hex_color)
        tcPr.append(shd)


def set_col_widths(table, widths: list[float]):
    for row in table.rows:
        for i, cell in enumerate(row.cells):
            if i < len(widths):
                cell.width = Inches(widths[i])


def add_heading(doc, text, level=1):
    h = doc.add_heading(text, level=level)
    h.style.font.color.rgb = RGBColor(0x1F, 0x49, 0x7D)
    return h


def add_table_header(table, headers: list[str]):
    hdr = table.rows[0]
    shade_row(hdr, "1F497D")
    for i, h in enumerate(headers):
        cell = hdr.cells[i]
        cell.text = h
        run = cell.paragraphs[0].runs[0]
        run.bold = True
        run.font.color.rgb = RGBColor(0xFF, 0xFF, 0xFF)
        run.font.size = Pt(8)
        cell.paragraphs[0].alignment = WD_ALIGN_PARAGRAPH.CENTER


def add_data_row(table, values: list, alt: bool = False):
    row = table.add_row()
    if alt:
        shade_row(row, "EBF3FB")
    for i, v in enumerate(values):
        cell = row.cells[i]
        cell.text = str(v)
        run = cell.paragraphs[0].runs[0] if cell.paragraphs[0].runs else cell.paragraphs[0].add_run(str(v))
        run.font.size = Pt(8)
        cell.paragraphs[0].alignment = WD_ALIGN_PARAGRAPH.CENTER
    return row


# ── load data ─────────────────────────────────────────────────────────────────

rows = []
with open(CSV_PATH) as f:
    for r in csv.DictReader(f):
        dist = float(r["ESTIMATED_DISTANCE"] or 0)
        ts_raw = r.get("TIME_STARTED", "").strip()
        try:
            ts = datetime.fromisoformat(ts_raw.replace(" Z", "+00:00"))
            elapsed_h = (NOW - ts).total_seconds() / 3600
        except Exception:
            elapsed_h = None
        route = json.loads(r["SELECTED_ROUTE"]) if r.get("SELECTED_ROUTE") else {}
        rows.append({
            "voyage_id": int(r["VOYAGE_ID"]),
            "start_port": r["START_PORT_ID"],
            "end_port":   r["END_PORT_ID"],
            "dist":       dist,
            "speed":      float(r["ESTIMATED_SPEED"] or 0),
            "route_dist": route.get("distance", 0),
            "route_dur":  route.get("duration", ""),
            "started":    ts_raw[:10] if ts_raw else "",
            "elapsed_h":  round(elapsed_h, 1) if elapsed_h else None,
            "same_port":  r["START_PORT_ID"] == r["END_PORT_ID"],
        })

zero  = sorted([r for r in rows if r["dist"] == 0],  key=lambda x: x["elapsed_h"] or 0, reverse=True)
micro = sorted([r for r in rows if r["dist"] > 0],   key=lambda x: x["dist"])

# bias fix results (from batch_audit run)
BIAS_FIX = {
    23149:  ("0.00h", "0.04h"),
    111720: ("0.00h", "0.14h"),
    96543:  ("0.00h", "0.52h"),
    125574: ("0.00h", "0.62h"),
    123199: ("0.00h", "0.79h"),
}


# ── build document ────────────────────────────────────────────────────────────

doc = Document()

# page margins
for section in doc.sections:
    section.top_margin    = Inches(0.9)
    section.bottom_margin = Inches(0.9)
    section.left_margin   = Inches(0.9)
    section.right_margin  = Inches(0.9)

# title
title = doc.add_heading("Prediction Discrepancy — Root Cause Report", 0)
title.alignment = WD_ALIGN_PARAGRAPH.CENTER

# metadata
meta = doc.add_paragraph()
meta.alignment = WD_ALIGN_PARAGRAPH.CENTER
meta.add_run("Date: 2026-10-25   |   Endpoint: opentug-live-predict-eta-py312   |   Voyages analysed: 43").font.size = Pt(9)

doc.add_paragraph()

# ── executive summary ─────────────────────────────────────────────────────────
add_heading(doc, "1. Executive Summary", 1)

p = doc.add_paragraph(
    "All 43 voyages previously returning consolidated_eta == as_of_utc (zero remaining hours) "
    "have been investigated end-to-end against live Snowflake data. The root cause splits into "
    "two categories. No fetch or inference bugs were found — the data in Snowflake is accurate "
    "and is being ingested correctly by the endpoint."
)
p.runs[0].font.size = Pt(10)

doc.add_paragraph()

# summary table
tbl = doc.add_table(rows=1, cols=4)
tbl.style = "Table Grid"
add_table_header(tbl, ["Category", "Count", "Root Cause", "Fixable by Inference?"])
set_col_widths(tbl, [1.5, 0.6, 3.5, 1.2])

add_data_row(tbl, ["Zero-distance holding assignments", "38",
                   "ESTIMATED_DISTANCE = 0, START_PORT_ID == END_PORT_ID — vessel moored, no transit",
                   "No — product/routing decision"], alt=False)
add_data_row(tbl, ["Micro-leg transit voyages", "5",
                   "Valid short trips (0.2–3.4 nm); flat 1.96h bias subtraction wiped their small predictions",
                   "Yes — Fixed in src/bias.py"], alt=True)

doc.add_paragraph()

# ── category 1 ────────────────────────────────────────────────────────────────
add_heading(doc, "2. Category 1 — 38 Zero-Distance Holding Assignments", 1)

p = doc.add_paragraph(
    "These voyages have ESTIMATED_DISTANCE = 0, SELECTED_ROUTE.distance = 0.0, "
    "SELECTED_ROUTE.duration = \"00:00:00\", and START_PORT_ID == END_PORT_ID. "
    "The vessel has never moved; TIME_COMPLETED is NULL on all 38. "
    "With no distance signal, Model A computes naive_h = 0 / speed = 0 and falls back "
    "to area priors, producing near-zero predictions (0.01–0.11h). "
    "These are not inference errors — the underlying trip has no destination."
)
p.runs[0].font.size = Pt(10)

doc.add_paragraph()

tbl2 = doc.add_table(rows=1, cols=6)
tbl2.style = "Table Grid"
add_table_header(tbl2, ["Voyage ID", "Port Pair", "Dist (nm)", "Speed (kn)", "Started", "Elapsed (h)"])
set_col_widths(tbl2, [0.85, 1.25, 0.85, 0.85, 1.0, 0.9])

for i, r in enumerate(zero):
    pp = f"{r['start_port']} → {r['end_port']}"
    add_data_row(tbl2, [
        r["voyage_id"], pp,
        f"{r['dist']:.3f}", f"{r['speed']:.2f}",
        r["started"], f"{r['elapsed_h']:,.1f}",
    ], alt=(i % 2 == 1))

doc.add_paragraph()

note = doc.add_paragraph()
note.add_run("Recommendation: ").bold = True
note.add_run(
    "Filter these voyages before inference using the condition "
    "ESTIMATED_DISTANCE = 0 OR START_PORT_ID = END_PORT_ID and return a "
    "harbor_operation status instead of running the model. "
    "The top 10 voyages by elapsed time have been open for 550–7,574 hours (23–315 days) "
    "without a single AIS movement or route update."
)
note.runs[0].font.size = Pt(9)
note.runs[1].font.size = Pt(9)

doc.add_paragraph()

# ── category 2 ────────────────────────────────────────────────────────────────
add_heading(doc, "3. Category 2 — 5 Micro-Leg Transit Voyages", 1)

p = doc.add_paragraph(
    "These five voyages have genuine planned routes with real multi-coordinate paths "
    "and non-zero distance (0.2–3.4 nm, 3–51 minutes planned transit). "
    "The previous flat 1.96-hour bias adjustment in src/bias.py fully wiped their "
    "predictions because max(raw - 1.96, 0.0) always evaluated to 0.0 for any trip "
    "shorter than 1.96 hours. The Snowflake route data confirms each trip is real."
)
p.runs[0].font.size = Pt(10)

doc.add_paragraph()

tbl3 = doc.add_table(rows=1, cols=8)
tbl3.style = "Table Grid"
add_table_header(tbl3, ["Voyage ID", "Port Pair", "Dist (nm)", "Speed (kn)",
                         "Route Duration", "Started", "Elapsed (h)", "Before → After Fix"])
set_col_widths(tbl3, [0.75, 1.1, 0.75, 0.75, 1.0, 0.9, 0.85, 1.15])

for i, r in enumerate(micro):
    before, after = BIAS_FIX.get(r["voyage_id"], ("—", "—"))
    pp = f"{r['start_port']} → {r['end_port']}"
    row = add_data_row(tbl3, [
        r["voyage_id"], pp,
        f"{r['dist']:.4f}", f"{r['speed']:.2f}",
        r["route_dur"][:8] if r["route_dur"] else "—",
        r["started"], f"{r['elapsed_h']:,.1f}",
        f"{before} → {after}",
    ], alt=(i % 2 == 1))

doc.add_paragraph()

note2 = doc.add_paragraph()
note2.add_run("Fix applied (src/bias.py): ").bold = True
note2.add_run(
    "Replaced max(raw - adj, 0.0) with effective_adj = min(adj, raw * 0.50); "
    "corr = max(raw - effective_adj, 0.0). "
    "The adjustment is now capped at 50% of the raw prediction so no non-zero "
    "prediction can be fully eliminated. All 5 voyages now return non-zero ETAs."
)
note2.runs[0].font.size = Pt(9)
note2.runs[1].font.size = Pt(9)

doc.add_paragraph()
note3 = doc.add_paragraph()
note3.add_run("Note on extreme elapsed times: ").bold = True
note3.add_run(
    "Voyages 23149 (10,004h open) and 96543 (2,980h open) produce small but non-zero "
    "remaining predictions because Model A is extrapolating far outside its training "
    "range for elapsed time. The predictions are numerically correct given the inputs; "
    "the underlying issue is that these voyages have been open for months without closure."
)
note3.runs[0].font.size = Pt(9)
note3.runs[1].font.size = Pt(9)

doc.add_paragraph()

# ── verification ──────────────────────────────────────────────────────────────
add_heading(doc, "4. Verification", 1)

bullets = [
    "43 / 43 payloads built from live Snowflake data without error.",
    "0 / 43 voyages return zero remaining hours after the bias fix (was 43 / 43 before).",
    "Snowflake schema confirmed: ESTIMATED_DISTANCE is the correct distance field. "
     "PLANNED_DIST_NM / PLANNED_DURATION_H / VESSEL_ID are not raw table columns — "
     "the inference pipeline already uses the correct field names.",
    "No fetch issue found. SELECT sub.* returns all relevant columns; missing columns "
     "in the audit script were incorrect expected names, not missing data.",
]
for b in bullets:
    p = doc.add_paragraph(style="List Bullet")
    p.add_run(b).font.size = Pt(10)

doc.add_paragraph()

# ── footer note ───────────────────────────────────────────────────────────────
footer_p = doc.add_paragraph()
footer_p.add_run(
    "Data source: EDPPROD.EDP_SOURCE.WEBATDS_SUBSERVICE queried 2026-10-25. "
    "Endpoint config: opentug-live-predict-config-20261025-121625 (deployed 2026-10-25 with bias fix)."
).font.size = Pt(8)
footer_p.alignment = WD_ALIGN_PARAGRAPH.CENTER

doc.save(OUT_PATH)
print(f"Saved: {OUT_PATH}")
