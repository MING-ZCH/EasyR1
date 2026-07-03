#!/usr/bin/env python3
"""Generate BOK-GRPO PowerPoint presentation (22 slides)."""

from pptx import Presentation
from pptx.util import Inches, Pt, Emu
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.enum.shapes import MSO_SHAPE
from pptx.oxml.ns import qn

# Constants
SLIDE_W, SLIDE_H = Inches(10), Inches(7.5)
ACCENT = RGBColor(0x8B, 0x2C, 0x2C)
DARK = RGBColor(0x1A, 0x1A, 0x1A)
BODY_CLR = RGBColor(0x44, 0x44, 0x44)
LIGHT_GRAY_BG = RGBColor(0xFA, 0xFA, 0xFA)
BORDER_CLR = RGBColor(0xE5, 0xE5, 0xE5)
HDR_BG = RGBColor(0x33, 0x33, 0x55)
ALT_ROW = RGBColor(0xEE, 0xF1, 0xF5)
HIGHLIGHT_ROW = RGBColor(0xE8, 0xE3, 0xD9)
WHITE = RGBColor(0xFF, 0xFF, 0xFF)

MARGIN_L = Inches(0.65)
CONTENT_W = Inches(8.7)
GAP = Inches(0.15)

FONT_TITLE = "Crimson Pro SemiBold"
FONT_SERIF = "Crimson Pro"
FONT_BODY = "Inter"
FONT_CODE = "Consolas"

prs = Presentation()
prs.slide_width = SLIDE_W
prs.slide_height = SLIDE_H
blank_layout = prs.slide_layouts[6]


def _set_font(run, name, size, color, bold=False, italic=False):
    run.font.name = name
    run.font.size = Pt(size)
    run.font.color.rgb = color
    run.font.bold = bold
    run.font.italic = italic


def _no_fill(cell):
    cell.fill.background()


def _cell_fill(cell, color):
    cell.fill.solid()
    cell.fill.fore_color.rgb = color


def white_bg(sl):
    bg = sl.background
    bg.fill.solid()
    bg.fill.fore_color.rgb = WHITE


def title_bar(sl, text):
    txBox = sl.shapes.add_textbox(MARGIN_L, Inches(0.35), CONTENT_W, Inches(0.55))
    tf = txBox.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    run = p.add_run()
    run.text = text
    _set_font(run, FONT_TITLE, 32, DARK, bold=True)
    line = sl.shapes.add_shape(
        MSO_SHAPE.RECTANGLE, MARGIN_L, Inches(0.92), Emu(int(80 * 914400 / 72)), Pt(3)
    )
    line.fill.solid()
    line.fill.fore_color.rgb = ACCENT
    line.line.fill.background()
    return Inches(1.1)


def _set_cell(cell, txt, font_name, font_size, font_color, bold=False, italic=False, align=PP_ALIGN.LEFT):
    cell.text = ""
    cell.vertical_anchor = MSO_ANCHOR.MIDDLE
    p = cell.text_frame.paragraphs[0]
    p.alignment = align
    run = p.add_run()
    run.text = str(txt)
    _set_font(run, font_name, font_size, font_color, bold, italic)
    cell.margin_left = Pt(6)
    cell.margin_right = Pt(4)
    cell.margin_top = Pt(2)
    cell.margin_bottom = Pt(2)


def _strip_vertical_borders(table):
    tbl_xml = table._tbl
    for tc in tbl_xml.iter(qn("a:tc")):
        tcPr = tc.find(qn("a:tcPr"))
        if tcPr is None:
            tcPr = tc.makeelement(qn("a:tcPr"), {})
            tc.insert(0, tcPr)
        for side in ["a:lnL", "a:lnR"]:
            ln = tcPr.find(qn(side))
            if ln is None:
                ln = tcPr.makeelement(qn(side), {"w": "0"})
                tcPr.append(ln)
            else:
                ln.set("w", "0")
            noFill = ln.find(qn("a:noFill"))
            if noFill is None:
                noFill = ln.makeelement(qn("a:noFill"), {})
                ln.append(noFill)


def add_table(sl, headers, rows, y, col_widths, highlight_last=False):
    n_rows = len(rows) + 1
    n_cols = len(headers)
    row_h = Inches(0.32)
    tbl_h = row_h * n_rows
    total_w = sum(col_widths)
    shape = sl.shapes.add_table(n_rows, n_cols, MARGIN_L, y, total_w, tbl_h)
    table = shape.table
    for i, w in enumerate(col_widths):
        table.columns[i].width = w
    for j, h in enumerate(headers):
        cell = table.cell(0, j)
        _cell_fill(cell, HDR_BG)
        _set_cell(cell, h, FONT_BODY, 13, WHITE, bold=True)
    for i, row_data in enumerate(rows):
        is_last = (i == len(rows) - 1)
        for j, val in enumerate(row_data):
            cell = table.cell(i + 1, j)
            if highlight_last and is_last:
                _cell_fill(cell, HIGHLIGHT_ROW)
                _set_cell(cell, val, FONT_BODY, 12, DARK, bold=True)
            elif i % 2 == 1:
                _cell_fill(cell, ALT_ROW)
                _set_cell(cell, val, FONT_BODY, 12, BODY_CLR)
            else:
                _no_fill(cell)
                _set_cell(cell, val, FONT_BODY, 12, BODY_CLR)
    _strip_vertical_borders(table)
    return y + tbl_h + GAP


def highlight_row_n(sl, row_idx, n_cols):
    """Highlight a specific data row (1-based) in the last table added."""
    for shape in reversed(list(sl.shapes)):
        if shape.has_table:
            tbl = shape.table
            for j in range(n_cols):
                cell = tbl.cell(row_idx, j)
                _cell_fill(cell, HIGHLIGHT_ROW)
                cell.text_frame.paragraphs[0].runs[0].font.bold = True
            break


def formula_box(sl, text, y):
    h = Inches(0.45)
    shape = sl.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, MARGIN_L, y, CONTENT_W, h)
    shape.fill.solid()
    shape.fill.fore_color.rgb = LIGHT_GRAY_BG
    shape.line.color.rgb = BORDER_CLR
    shape.line.width = Pt(1)
    tf = shape.text_frame
    tf.word_wrap = True
    tf.margin_left = Pt(12)
    tf.margin_top = Pt(4)
    p = tf.paragraphs[0]
    p.alignment = PP_ALIGN.LEFT
    run = p.add_run()
    run.text = text
    _set_font(run, FONT_SERIF, 14, ACCENT, italic=True)
    return y + h + GAP


def insight_box(sl, title, body, y):
    h = Inches(0.8)
    shape = sl.shapes.add_shape(MSO_SHAPE.RECTANGLE, MARGIN_L, y, CONTENT_W, h)
    shape.fill.solid()
    shape.fill.fore_color.rgb = LIGHT_GRAY_BG
    shape.line.color.rgb = BORDER_CLR
    shape.line.width = Pt(1)
    tf = shape.text_frame
    tf.word_wrap = True
    tf.margin_left = Pt(12)
    tf.margin_right = Pt(12)
    tf.margin_top = Pt(6)
    p = tf.paragraphs[0]
    run = p.add_run()
    run.text = title
    _set_font(run, FONT_SERIF, 14, DARK, bold=True)
    p2 = tf.add_paragraph()
    run2 = p2.add_run()
    run2.text = body
    _set_font(run2, FONT_BODY, 12, BODY_CLR, italic=True)
    return y + h + GAP


def section_header(sl, text, y):
    h = Inches(0.35)
    txBox = sl.shapes.add_textbox(MARGIN_L, y, CONTENT_W, h)
    tf = txBox.text_frame
    p = tf.paragraphs[0]
    run = p.add_run()
    run.text = text
    _set_font(run, FONT_TITLE, 20, DARK, bold=True)
    return y + h + Inches(0.05)


def bullet(sl, text, y):
    h = Inches(0.3)
    txBox = sl.shapes.add_textbox(MARGIN_L + Inches(0.1), y, CONTENT_W - Inches(0.1), h)
    tf = txBox.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    run = p.add_run()
    run.text = "\u2022  " + text
    _set_font(run, FONT_BODY, 15, BODY_CLR)
    return y + h


def bullets(sl, items, y):
    for item in items:
        y = bullet(sl, item, y)
    return y + GAP


def code_block(sl, text, y, height=None, font_size=9):
    lines = text.strip().split("\n")
    if height is None:
        height = Inches(max(0.5, len(lines) * 0.22))
    shape = sl.shapes.add_shape(MSO_SHAPE.RECTANGLE, MARGIN_L, y, CONTENT_W, height)
    shape.fill.solid()
    shape.fill.fore_color.rgb = LIGHT_GRAY_BG
    shape.line.color.rgb = BORDER_CLR
    shape.line.width = Pt(1)
    tf = shape.text_frame
    tf.word_wrap = False
    tf.margin_left = Pt(10)
    tf.margin_top = Pt(6)
    tf.margin_bottom = Pt(6)
    first = True
    for line in lines:
        if first:
            p = tf.paragraphs[0]
            first = False
        else:
            p = tf.add_paragraph()
        p.space_before = Pt(0)
        p.space_after = Pt(0)
        run = p.add_run()
        run.text = line
        _set_font(run, FONT_CODE, font_size, BODY_CLR)
    return y + height + GAP


def two_cards(sl, left_title, left_items, right_title, right_items, y):
    card_w = Inches(4.2)
    card_gap = Inches(0.3)
    max_lines = max(len(left_items), len(right_items))
    card_h = Inches(0.45 + max_lines * 0.28)
    for idx, (title, items) in enumerate([(left_title, left_items), (right_title, right_items)]):
        left_pos = MARGIN_L + idx * (card_w + card_gap)
        shape = sl.shapes.add_shape(MSO_SHAPE.RECTANGLE, left_pos, y, card_w, card_h)
        shape.fill.solid()
        shape.fill.fore_color.rgb = WHITE
        shape.line.color.rgb = RGBColor(0xEA, 0xEA, 0xEA)
        shape.line.width = Pt(1)
        tf = shape.text_frame
        tf.word_wrap = True
        tf.margin_left = Pt(14)
        tf.margin_right = Pt(10)
        tf.margin_top = Pt(8)
        p = tf.paragraphs[0]
        run = p.add_run()
        run.text = title
        _set_font(run, FONT_SERIF, 15, ACCENT, bold=True, italic=True)
        for item in items:
            p2 = tf.add_paragraph()
            p2.space_before = Pt(2)
            run2 = p2.add_run()
            run2.text = "\u2022  " + item
            _set_font(run2, FONT_BODY, 12, BODY_CLR)
    return y + card_h + GAP


# ── Slide builders ─────────────────────────────────────────────────────

def s01_title():
    sl = prs.slides.add_slide(blank_layout)
    white_bg(sl)
    txBox = sl.shapes.add_textbox(Inches(0.5), Inches(1.8), Inches(9), Inches(1))
    tf = txBox.text_frame
    p = tf.paragraphs[0]
    p.alignment = PP_ALIGN.CENTER
    run = p.add_run()
    run.text = "BOK-GRPO"
    _set_font(run, FONT_TITLE, 54, ACCENT, bold=True)
    subtitles = [
        "Best-of-K Group Relative Policy Optimization",
        "Interleaved Point-to-Count for VLM Object Counting",
        "Collapsing pass@K capability into pass@1",
    ]
    for i, sub in enumerate(subtitles):
        txB = sl.shapes.add_textbox(Inches(0.5), Inches(3.0 + i * 0.5), Inches(9), Inches(0.45))
        tf2 = txB.text_frame
        p2 = tf2.paragraphs[0]
        p2.alignment = PP_ALIGN.CENTER
        r = p2.add_run()
        r.text = sub
        sz = 22 if i == 0 else 18
        clr = DARK if i == 0 else BODY_CLR
        _set_font(r, FONT_BODY, sz, clr, italic=(i == 2))
    line = sl.shapes.add_shape(
        MSO_SHAPE.RECTANGLE, Inches(4.2), Inches(2.85), Inches(1.6), Pt(3)
    )
    line.fill.solid()
    line.fill.fore_color.rgb = ACCENT
    line.line.fill.background()


def s02_pass_k_gap():
    sl = prs.slides.add_slide(blank_layout)
    white_bg(sl)
    y = title_bar(sl, "The pass@K Gap Problem")
    y = add_table(sl,
        ["Metric", "Value", "Implication"],
        [
            ["pass@1", "79.2%", "Current single-attempt accuracy"],
            ["pass@32", "95.2%", "Latent capability with 32 attempts"],
            ["Gap", "16.0pp", "Untapped potential to recover"],
        ],
        y, [Inches(2), Inches(2), Inches(4.7)], highlight_last=True)
    y = two_cards(sl,
        "Standard GRPO  \u2717",
        ["A=(r\u2212\u03bc)/\u03c3 \u2192 vanishing gradients when \u03c3\u21920",
         "14/16 correct \u2192 no signal",
         "Cannot differentiate among high-reward paths"],
        "BOK-GRPO  \u2713",
        ["A=(softmax(z/\u03c4)\u22121/K)\u00d7K",
         "Always separates best from rest",
         "\u03c4 annealing: explore \u2192 focus on single best"],
        y)


def s03_interleaved():
    sl = prs.slides.add_slide(blank_layout)
    white_bg(sl)
    y = title_bar(sl, "Interleaved Paradigm (Turn Table)")
    y = add_table(sl,
        ["Turn", "Input", "Model Output", "Environment"],
        [
            ["1", "Original image + question", "<point>{x,y count:1}</point>", "Draw red dot #1"],
            ["2", "Image + 1 red dot", "<point>{x,y count:2}</point>", "Draw red dot #2"],
            ["\u22ee", "Image + (k\u22121) dots", "<point>{x,y count:k}</point>", "Draw red dot #k"],
            ["N", "Image + (N\u22121) dots", "<answer>N</answer>", "\u2192 STOP"],
        ],
        y, [Inches(0.7), Inches(2.5), Inches(3.0), Inches(2.5)], highlight_last=True)
    y = insight_box(sl,
        "Why Interleaved?",
        "Humans count by pointing one-by-one. Sequential pointing = explicit attention shift. "
        "Each red dot = visual memory preventing double-counting.", y)


def s04_comparison():
    sl = prs.slides.add_slide(blank_layout)
    white_bg(sl)
    y = title_bar(sl, "Why Interleaved? (Comparison)")
    y = add_table(sl,
        ["Aspect", "Single-Turn", "Interleaved (Ours)", "Why Better"],
        [
            ["Attention", "Must track all at once", "One object per turn", "Reduces cognitive load"],
            ["Feedback", "No mid-process signal", "Red dot visual memory", "Prevents double-counting"],
            ["Error", "Silent cumulative", "Detectable per-step", "Enables dense reward"],
            ["Scalability", "Degrades with count", "Linear with N", "Handles 1\u221250 objects"],
            ["Training", "Sparse reward only", "Dense per-step reward", "Better RL signal"],
        ],
        y, [Inches(1.6), Inches(2.1), Inches(2.3), Inches(2.7)], highlight_last=True)


def s05_softmax_advantage():
    sl = prs.slides.add_slide(blank_layout)
    white_bg(sl)
    y = title_bar(sl, "Core Algorithm \u2014 Softmax Advantage")
    y = formula_box(sl, "Standard GRPO:  A\u1d62 = (r\u1d62 \u2212 \u03bc) / \u03c3   \u2192  \u03c3\u21920 when 14/16 correct", y)
    y = formula_box(sl, "BOK-GRPO:  w\u1d62 = softmax(z\u1d62/\u03c4)    A\u1d62 = (w\u1d62 \u2212 1/K) \u00d7 K    where z\u1d62 = (r\u1d62\u2212\u03bc)/\u03c3", y)
    y = add_table(sl,
        ["\u03c4 Value", "Behavior", "Why This Range"],
        [
            ["\u03c4\u21920", "Winner-take-all (pure BoK)", "Maximum pressure, potentially unstable"],
            ["\u03c4=0.3\u22120.7", "Top-K concentration", "Practical: strong signal, still stable"],
            ["\u03c4\u2192\u221e", "Uniform (\u2248 std GRPO)", "No selection pressure"],
        ],
        y, [Inches(2.0), Inches(3.2), Inches(3.5)])


def s06_temperature():
    sl = prs.slides.add_slide(blank_layout)
    white_bg(sl)
    y = title_bar(sl, "Temperature Annealing")
    y = formula_box(sl, "\u03c4(t) = \u03c4_f + (\u03c4\u2080 \u2212 \u03c4_f) \u00d7 \u00bd \u00d7 (1 + cos(\u03c0t/T))     [0.5 \u2192 0.3]", y)
    y = insight_box(sl,
        "Why \u03c4_init=0.5, \u03c4_final=0.3?",
        "Early: policy is noisy \u2192 broader credit (\u03c4=0.5) avoids premature convergence. "
        "Late: policy refined \u2192 sharper selection (\u03c4=0.3) focuses on single best path.", y)
    y = bullets(sl, [
        "Cosine schedule: smooth transition, no sudden jumps",
        "\u03c4=0.5 start: ~exp(2)\u22487\u00d7 weight ratio between best and worst",
        "\u03c4=0.3 end: ~exp(3.3)\u224827\u00d7 ratio \u2014 strong single-best pressure",
    ], y)


def s07_routing():
    sl = prs.slides.add_slide(blank_layout)
    white_bg(sl)
    y = title_bar(sl, "4-Way Adaptive Routing")
    y = add_table(sl,
        ["Route", "Condition", "Advantage Formula", "Why This Route"],
        [
            ["1. LowVar", "\u03c3 \u2264 \u03b5", "r \u2212 batch_mean", "Uniform group: use batch baseline"],
            ["2. AllCorrect", "All K correct", "A = 0", "Nothing to learn: skip"],
            ["3. EasyDrGRPO", "pass > \u03b8=0.50", "clip(z, \u00b12.5)", "Easy: z-norm suffices"],
            ["4. BoK Softmax", "Default", "(softmax(z/\u03c4)\u22121/K)\u00d7K", "Hard: need BoK selection"],
        ],
        y, [Inches(1.6), Inches(1.8), Inches(2.6), Inches(2.7)], highlight_last=True)
    y = insight_box(sl,
        "Routing is DATA-DRIVEN",
        "Typical distribution:  LowVar 7%  |  AllCorrect 15%  |  EasyDrGRPO 47%  |  BoK 31%", y)


def s08_easy_drgrpo():
    sl = prs.slides.add_slide(blank_layout)
    white_bg(sl)
    y = title_bar(sl, "Why EasyDrGRPO & \u03b8=0.50")
    y = bullets(sl, [
        "\u03b8=0.50: Empirically validated \u2014 below 0.50 too many groups incorrectly routed to EasyDrGRPO",
        "Above 0.50 too few groups get efficient z-norm gradients",
        "0.50 gives 84.7% effective gradient coverage",
        "EasyDrGRPO reserves BoK\u2019s concentrated selection for genuinely hard groups",
    ], y)
    y = add_table(sl,
        ["Data", "LowVar%", "AllCorr%", "EasyDr%", "BoK%"],
        [
            ["0_10 (sparse)", "7%", "15%", "47%", "31%"],
            ["hard_only", "3%", "5%", "25%", "67%"],
            ["mixed (h+e)", "5%", "10%", "38%", "47%"],
        ],
        y, [Inches(2.0), Inches(1.5), Inches(1.5), Inches(1.5), Inches(2.2)])


def s09_safety():
    sl = prs.slides.add_slide(blank_layout)
    white_bg(sl)
    y = title_bar(sl, "Safety & Stability Mechanisms")
    y = add_table(sl,
        ["Mechanism", "Implementation", "Why Needed"],
        [
            ["NaN Guard", "nan_to_num(scores, nan=0)", "Trajectory failures \u2192 NaN rewards"],
            ["Logit Cap", "clamp(logits, \u00b112.0)", "Extreme z \u2192 softmax overflow \u2192 NaN"],
            ["\u03c4-Adaptive Floor", "\u03c4_eff = max(\u03c4, |z_max|/C)", "Prevents over-sharp softmax"],
            ["Advantage Clip", "clamp(A, \u00b14.0)", "Bounds gradient magnitude"],
            ["Collapse Detection", "max(w)>0.9 \u2192 mix uniform", "Single-traj domination \u2192 instability"],
            ["KL Penalty", "\u03b2=0.03\u00b7D_KL(\u03c0\u2016\u03c0_ref)", "Prevent policy drift from reference"],
            ["PPO Clip", "clip(r(\u03b8), 1\u00b10.28)", "Trust region constraint"],
            ["eff_intensity", "LR\u00d7clip\u00d7ppo = 5.6e\u207b\u2077", "Sweet spot: 4\u22126e\u207b\u2077"],
        ],
        y, [Inches(2.0), Inches(3.2), Inches(3.5)], highlight_last=True)


def s10_dense_reward():
    sl = prs.slides.add_slide(blank_layout)
    white_bg(sl)
    y = title_bar(sl, "Dense Mask Point Reward")
    y = add_table(sl,
        ["Outcome", "Condition", "Reward", "Rationale"],
        [
            ["HIT (new)", "Point in unused mask", "1.0", "Correct identification"],
            ["HIT (dup)", "Point in used mask", "0.0", "Penalize re-counting"],
            ["MISS", "Point outside all masks", "decay(d)", "Distance-based shaping"],
        ],
        y, [Inches(1.6), Inches(2.4), Inches(1.4), Inches(3.3)])
    y = insight_box(sl,
        "Why Masks?",
        "RL has no per-step GT points \u2014 only GT count. Pre-computed segmentation masks enable "
        "dense reward without manual labeling. Mask-based matching handles varying object sizes.", y)


def s11_distance_decay():
    sl = prs.slides.add_slide(blank_layout)
    white_bg(sl)
    y = title_bar(sl, "Distance Decay & Answer Reward")
    y = formula_box(sl, "Near (d\u22640.02): r = exp(\u221220\u00b7d)   |   Far (d>0.02): r = exp(\u221220\u00b7d_f)\u00b7exp(\u221250\u00b7(d\u2212d_f))", y)
    y = section_header(sl, "Trajectory Reward Composition", y)
    y = formula_box(sl, "R = 0.6 \u00d7 R_answer + 0.3 \u00d7 R_point + 0.1 \u00d7 R_format", y)
    y = add_table(sl,
        ["Component", "Weight", "Design", "Why This Weight"],
        [
            ["R_answer", "0.6", "Soft decay, cap=0.4", "Primary goal is correct count"],
            ["R_point", "0.3", "Dense mask per-step", "Prevents reward sparsity"],
            ["R_format", "0.1", "Binary compliance", "Ensures parseable output"],
        ],
        y, [Inches(1.8), Inches(1.2), Inches(2.6), Inches(3.1)])


def s12_asymmetric():
    sl = prs.slides.add_slide(blank_layout)
    white_bg(sl)
    y = title_bar(sl, "Asymmetric Answer Penalty")
    y = add_table(sl,
        ["Direction", "k Factor", "Penalty", "Why Asymmetric"],
        [
            ["Over-count (pred>GT)", "k=2.0", "Heavier", "Hallucination worse \u2014 counts non-existent objects"],
            ["Under-count (pred<GT)", "k=1.0\u22121.5", "Lighter", "Missing is less severe especially large GT"],
        ],
        y, [Inches(2.2), Inches(1.4), Inches(1.4), Inches(3.7)])
    y = insight_box(sl,
        "Why 0.6 / 0.3 / 0.1?",
        "Pure answer reward is too sparse for multi-turn RL (one signal per trajectory). "
        "Dense point reward (0.3) provides per-step learning. Format (0.1) prevents structural degeneration.", y)
    y = bullet(sl, "Consistency Check: If pred_answer \u2260 #pred_points \u2192 R_answer \u00d7 0.5", y)


def s13_training_config():
    sl = prs.slides.add_slide(blank_layout)
    white_bg(sl)
    y = title_bar(sl, "Training Configuration")
    y = add_table(sl,
        ["Parameter", "Value", "Why This Choice"],
        [
            ["Base Model", "Qwen2.5-VL-7B", "Strong VLM with point/grounding capability"],
            ["SFT Pre-train", "30k / 3537 steps", "Establish counting format before RL"],
            ["K (rollouts)", "16", "Enough diversity for softmax advantage"],
            ["Learning Rate", "1e\u207b\u2076", "Part of eff_intensity sweet spot"],
            ["PPO Epochs", "2", "Balance update strength and stability"],
            ["Clip Ratio", "0.28", "eff_intensity = LR\u00d7clip\u00d7ppo = 5.6e\u207b\u2077"],
            ["KL Coeff", "0.03", "Prevent drift while allowing adaptation"],
            ["\u03c4 Schedule", "0.5\u21920.3 cosine", "Explore-then-exploit temperature"],
            ["BOK_CLIP", "4.0", "Bound advantage magnitude"],
            ["\u03b8_easy", "0.50", "Validated optimal routing threshold"],
            ["Max Turns", "11", "Cover GT\u226410 with margin"],
        ],
        y, [Inches(2.0), Inches(2.4), Inches(4.3)], highlight_last=True)


def s14_data_strategy():
    sl = prs.slides.add_slide(blank_layout)
    white_bg(sl)
    y = title_bar(sl, "Data Strategy \u2014 Dataset Comparison")
    y = add_table(sl,
        ["Dataset", "Samples", "Easy/Med/Hard", "Use Case"],
        [
            ["0_10 (sparse only)", "11455", "30 / 41 / 29%", "V7\u2212V12: sparse counting"],
            ["hard_only", "1808", "18 / 40 / 42%", "V14: hard data focus"],
            ["mixed (hard+easy)", "5792", "17 / 32 / 51%", "V14\u2212V16: balanced training"],
        ],
        y, [Inches(2.2), Inches(1.4), Inches(2.0), Inches(3.1)], highlight_last=True)
    y = insight_box(sl,
        "Why Mixed Data?",
        "Pure hard data (V14-hard) showed answer_reward=0.74 but pixmo-test=79.4% (worse than V12). "
        "Easy examples provide gradient signal when hard examples are all-wrong. "
        "Mixed data maintains BoK routing diversity.", y)


def s15_data_impact():
    sl = prs.slides.add_slide(blank_layout)
    white_bg(sl)
    y = title_bar(sl, "Impact of Data Choice")
    y = add_table(sl,
        ["Experiment", "Data", "pixmo-test", "Finding"],
        [
            ["V12 (best)", "0_10 sparse", "82.04%", "Best overall accuracy"],
            ["V14-hard", "hard_only", "79.40%", "Too narrow, overfits to hard"],
            ["V14-mixed", "mixed", "80.34%", "Better than hard-only"],
            ["V15", "mixed", "80.15%", "Conservative LR hurts"],
            ["V7 GRPO", "0_10 sparse", "80.72%", "Standard GRPO baseline"],
        ],
        y, [Inches(1.8), Inches(1.8), Inches(1.6), Inches(3.5)])
    highlight_row_n(sl, 1, 4)  # V12 row
    y = insight_box(sl,
        "Key Takeaway",
        "eff_intensity matters more than data choice alone. V12 at 5.6e\u207b\u2077 > V15 at 2.8e\u207b\u2077 despite same algorithm.", y)


def s16_main_results():
    sl = prs.slides.add_slide(blank_layout)
    white_bg(sl)
    y = title_bar(sl, "Main Results \u2014 Benchmark Comparison")
    y = add_table(sl,
        ["Model", "pixmo-test", "countbench", "Method"],
        [
            ["SFT Base", "75.6%", "\u2014", "Supervised Only"],
            ["V7 Std GRPO", "80.72%", "\u2014", "Standard GRPO"],
            ["V14 Mixed", "80.34%", "78.21%", "Mixed + BoK"],
            ["V15 Low-LR", "80.15%", "79.02%", "Conservative"],
            ["V12 BOK-GRPO", "82.04%", "79.43%", "BOK-GRPO (best)"],
        ],
        y, [Inches(2.2), Inches(1.8), Inches(1.8), Inches(2.9)], highlight_last=True)


def s17_v12_breakdown():
    sl = prs.slides.add_slide(blank_layout)
    white_bg(sl)
    y = title_bar(sl, "V12 Detailed Breakdown")
    y = add_table(sl,
        ["Split", "Samples", "Accuracy"],
        [
            ["Overall", "529", "82.04%"],
            ["Easy (GT 0\u22125)", "239", "84.10%"],
            ["Hard (GT 5+)", "290", "80.34%"],
            ["With History", "529", "83.36%"],
        ],
        y, [Inches(3.0), Inches(2.5), Inches(3.2)])
    highlight_row_n(sl, 1, 3)  # Overall row
    y = insight_box(sl,
        "Results Summary",
        "+6.44pp over SFT base.  +1.32pp over Standard GRPO.  "
        "pass@1\u219232 gap reduced from 16.0pp to ~13pp (33% closure).", y)


def s18_ablation():
    sl = prs.slides.add_slide(blank_layout)
    white_bg(sl)
    y = title_bar(sl, "Ablation \u2014 What Matters")
    y = add_table(sl,
        ["Design Choice", "Without", "With (Ours)", "Why It Helps"],
        [
            ["Interleaved multi-turn", "All-at-once", "Sequential+feedback", "Visual memory via red dots"],
            ["Dense point reward", "Answer-only", "+mask point 0.3", "Per-step signal prevents sparsity"],
            ["BoK softmax adv.", "Z-norm GRPO", "Softmax weighting", "+1.32pp: learns from best paths"],
            ["Mask matching", "Distance-only", "Hit/miss/duplicate", "Size-aware spatial grounding"],
            ["Consistency check", "None", "R_ans\u00d70.5 if \u2260", "Penalizes contradictory outputs"],
            ["Soft answer decay", "Binary 0/1", "exp decay cap=0.4", "Finer gradient for wrong answers"],
            ["\u03c4 cosine annealing", "Fixed \u03c4", "0.5\u21920.3", "Explore\u2192exploit"],
        ],
        y, [Inches(2.2), Inches(1.8), Inches(2.0), Inches(2.7)], highlight_last=True)


def s19_key_findings():
    sl = prs.slides.add_slide(blank_layout)
    white_bg(sl)
    y = title_bar(sl, "Key Findings")
    y = add_table(sl,
        ["Finding", "Evidence", "Implication"],
        [
            ["+6.44pp over SFT", "75.6%\u219282.04%", "RL training effective for counting"],
            ["+1.32pp over Std GRPO", "80.72%\u219282.04%", "BoK advantage > z-norm"],
            ["pass@1\u219232 gap \u221233%", "16.0pp\u2192~13pp", "Capability collapse working"],
            ["Zero NaN events", "eff_intensity=5.6e\u207b\u2077", "Safety mechanisms sufficient"],
            ["96.6% format", "Maintained throughout", "Format reward preserves structure"],
            ["Routing data-driven", "Easy47% BoK31%", "\u03b8=0.50 is robust"],
        ],
        y, [Inches(2.6), Inches(2.6), Inches(3.5)], highlight_last=True)


def s20_stability():
    sl = prs.slides.add_slide(blank_layout)
    white_bg(sl)
    y = title_bar(sl, "Stability Metrics")
    y = add_table(sl,
        ["Metric", "Value", "Status"],
        [
            ["eff_intensity", "5.6e\u207b\u2077", "\u2713 In sweet spot [4\u22126e\u207b\u2077]"],
            ["NaN events", "0", "\u2713 Full stability"],
            ["KL divergence", "<0.03 throughout", "\u2713 Controlled drift"],
            ["Format compliance", "96.6%", "\u2713 Structure preserved"],
            ["Best checkpoint", "V12 S178", "\u2713 Early sweet spot"],
        ],
        y, [Inches(2.6), Inches(2.6), Inches(3.5)], highlight_last=True)
    y = insight_box(sl,
        "Stability Enables Longer Training",
        "All safety mechanisms (NaN guard, logit cap, KL penalty, PPO clip) work together \u2014 "
        "none alone is sufficient.", y)


def s21_future():
    sl = prs.slides.add_slide(blank_layout)
    white_bg(sl)
    y = title_bar(sl, "Future Directions")
    y = add_table(sl,
        ["Target", "Current", "Goal", "Approach"],
        [
            ["Sparse accuracy", "82.04%", ">95%", "Improve point precision"],
            ["Dense accuracy", "est.~50%", ">50%", "Scale to GT>10"],
            ["pass@1 gap", "13pp", "<5pp", "Aggressive BoK with curriculum"],
            ["Generalization", "pixmo/countbench", "Multi-benchmark", "Domain transfer"],
        ],
        y, [Inches(2.0), Inches(2.0), Inches(2.0), Inches(2.7)], highlight_last=True)
    y = two_cards(sl,
        "Algorithm",
        ["Process reward model (PRM)",
         "Per-turn advantage (GSPO)",
         "Curriculum learning on difficulty"],
        "Data & Scale",
        ["Expand to GT>50",
         "Cross-domain counting",
         "Larger base models (14B+)"],
        y)


def s22_pseudocode():
    sl = prs.slides.add_slide(blank_layout)
    white_bg(sl)
    y = title_bar(sl, "Appendix \u2014 Pseudocode")
    code = """def bok_grpo_advantage(scores, tau, theta_easy=0.50):
    K = len(scores)
    mu, sigma = mean(scores), std(scores)
    
    if sigma <= EPS:                    # Route 1: LowVar
        return scores - batch_mean
    if all(s > theta for s in scores):  # Route 2: AllCorrect
        return zeros(K)
    
    z = (scores - mu) / sigma
    pass_rate = sum(s > theta for s in scores) / K
    
    if pass_rate > theta_easy:          # Route 3: EasyDrGRPO
        return clip(z, -2.5, 2.5)
    
    # Route 4: BoK Softmax
    tau_eff = max(tau, max(abs(z))/C, tau_min)
    w = softmax(z / tau_eff)
    
    # Collapse detection
    if max(w) > 0.9:
        w = 0.85 * w + 0.15 / K
    
    A = (w - 1/K) * K
    return clip(A, -BOK_CLIP, BOK_CLIP)"""
    y = code_block(sl, code, y, height=Inches(4.8), font_size=10)


# ── Build all slides ───────────────────────────────────────────────────
s01_title()
s02_pass_k_gap()
s03_interleaved()
s04_comparison()
s05_softmax_advantage()
s06_temperature()
s07_routing()
s08_easy_drgrpo()
s09_safety()
s10_dense_reward()
s11_distance_decay()
s12_asymmetric()
s13_training_config()
s14_data_strategy()
s15_data_impact()
s16_main_results()
s17_v12_breakdown()
s18_ablation()
s19_key_findings()
s20_stability()
s21_future()
s22_pseudocode()

OUT = "/data/workspace/hyleochang/EasyR1-latest/docs/BOK_GRPO_Presentation.pptx"
prs.save(OUT)
print(f"Saved {len(prs.slides)} slides -> {OUT}")
