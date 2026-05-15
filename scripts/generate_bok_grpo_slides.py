#!/usr/bin/env python3
"""BOK-GRPO Presentation — Academic Style with Design Rationale ("Why").

Flow: Motivation → Method → Experiments → Findings
Every design choice includes rationale explanation.
"""

from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.enum.text import PP_ALIGN
from pptx.dml.color import RGBColor

# ── Colors ──
C_PRIMARY   = RGBColor(0x1A, 0x1A, 0x1A)
C_SECONDARY = RGBColor(0x55, 0x55, 0x55)
C_ACCENT    = RGBColor(0x8B, 0x2C, 0x2C)
C_BORDER    = RGBColor(0xEA, 0xEA, 0xEA)
C_ROW_ALT   = RGBColor(0xEE, 0xF1, 0xF5)
C_ROW_FINAL = RGBColor(0xE8, 0xE3, 0xD9)
C_BOX_BG    = RGBColor(0xFA, 0xFA, 0xFA)
C_WHITE     = RGBColor(0xFF, 0xFF, 0xFF)
C_DIVIDER   = RGBColor(0xE5, 0xE5, 0xE5)
C_BODY      = RGBColor(0x44, 0x44, 0x44)
F_S, F_N = 'Crimson Pro', 'Inter'

def wb(sl):
    sl.background.fill.solid(); sl.background.fill.fore_color.rgb = C_WHITE

def rl(sl, l, t, w=0.8):
    s = sl.shapes.add_shape(1, l, t, Inches(w), Pt(3))
    s.fill.solid(); s.fill.fore_color.rgb = C_ACCENT; s.line.fill.background()

def tb(sl, txt, top=Inches(0.4)):
    b = sl.shapes.add_textbox(Inches(0.6), top, Inches(8.5), Inches(0.65))
    p = b.text_frame.paragraphs[0]; b.text_frame.word_wrap = True
    p.text = txt; p.font.size = Pt(36); p.font.name = F_S
    p.font.bold = True; p.font.color.rgb = C_PRIMARY
    rl(sl, Inches(0.6), top + Inches(0.6))
    return top + Inches(0.9)

def sec(sl, txt, y, l=Inches(0.6)):
    b = sl.shapes.add_textbox(l, y, Inches(8.5), Inches(0.4))
    p = b.text_frame.paragraphs[0]; b.text_frame.word_wrap = True
    p.text = txt; p.font.size = Pt(22); p.font.name = F_S
    p.font.bold = True; p.font.color.rgb = C_PRIMARY
    return y + Inches(0.48)

def bul(sl, txt, y, l=Inches(0.85), sz=Pt(16), c=C_BODY, w=Inches(8.3)):
    b = sl.shapes.add_textbox(l, y, w, Inches(0.32))
    b.text_frame.word_wrap = True; p = b.text_frame.paragraphs[0]
    p.text = txt; p.font.size = sz; p.font.name = F_N; p.font.color.rgb = c
    return y + Inches(0.34)

def fm(sl, txt, y, l=Inches(0.85), w=Inches(8.3)):
    b = sl.shapes.add_textbox(l, y, w, Inches(0.42))
    b.fill.solid(); b.fill.fore_color.rgb = C_BOX_BG
    b.line.color.rgb = C_BORDER; b.line.width = Pt(1)
    b.text_frame.word_wrap = True; b.text_frame.margin_left = Inches(0.1)
    p = b.text_frame.paragraphs[0]
    p.text = txt; p.font.size = Pt(16); p.font.name = F_S
    p.font.italic = True; p.font.color.rgb = C_ACCENT; p.alignment = PP_ALIGN.CENTER
    return y + Inches(0.52)

def ib(sl, title, body, y, l=Inches(0.6), w=Inches(8.8)):
    h = Inches(0.75)
    b = sl.shapes.add_textbox(l, y, w, h)
    b.fill.solid(); b.fill.fore_color.rgb = C_BOX_BG
    b.line.color.rgb = C_DIVIDER; b.line.width = Pt(1)
    tf = b.text_frame; tf.word_wrap = True
    tf.margin_left = Inches(0.15); tf.margin_top = Inches(0.08)
    p1 = tf.paragraphs[0]
    p1.text = title; p1.font.size = Pt(16); p1.font.name = F_S
    p1.font.bold = True; p1.font.color.rgb = C_PRIMARY
    p2 = tf.add_paragraph()
    p2.text = body; p2.font.size = Pt(14); p2.font.name = F_N
    p2.font.italic = True; p2.font.color.rgb = C_SECONDARY
    return y + h + Inches(0.1)

def tbl(sl, hd, rows, y, l=Inches(0.6), w=Inches(8.8), cw=None, hl=False):
    nr, nc = len(rows)+1, len(hd)
    ts = sl.shapes.add_table(nr, nc, l, y, w, Inches(0.36*nr))
    t = ts.table
    if cw:
        for i,v in enumerate(cw): t.columns[i].width = Inches(v)
    for j,h in enumerate(hd):
        c = t.cell(0,j); c.text = h; c.fill.solid(); c.fill.fore_color.rgb = C_WHITE
        for p in c.text_frame.paragraphs:
            p.font.size = Pt(16); p.font.bold = True; p.font.name = F_N
            p.font.color.rgb = C_PRIMARY
            p.alignment = PP_ALIGN.LEFT if j==0 else PP_ALIGN.CENTER
    for i,row in enumerate(rows):
        last = (i==len(rows)-1) and hl
        for j,val in enumerate(row):
            c = t.cell(i+1,j); c.text = str(val); c.fill.solid()
            c.fill.fore_color.rgb = C_ROW_FINAL if last else (C_ROW_ALT if i%2==1 else C_WHITE)
            for p in c.text_frame.paragraphs:
                p.font.size = Pt(16); p.font.name = F_N
                p.font.color.rgb = C_ACCENT if last else C_BODY
                p.font.bold = last
                p.alignment = PP_ALIGN.LEFT if j==0 else PP_ALIGN.CENTER
    return y + Inches(0.36*nr) + Inches(0.12)

def cb(sl, txt, y, l=Inches(0.6), w=Inches(8.8), h=Inches(5.0), fs=Pt(13)):
    b = sl.shapes.add_textbox(l, y, w, h)
    b.fill.solid(); b.fill.fore_color.rgb = C_BOX_BG
    b.line.color.rgb = C_BORDER; b.line.width = Pt(1)
    tf = b.text_frame; tf.word_wrap = True
    tf.margin_left = Inches(0.2); tf.margin_top = Inches(0.12)
    for i,ln in enumerate(txt.strip().split('\n')):
        p = tf.paragraphs[0] if i==0 else tf.add_paragraph()
        p.text = ln; p.font.size = fs; p.font.name = 'Consolas'
        s = ln.strip()
        if s.startswith('#') or s.startswith('//'):
            p.font.color.rgb = C_SECONDARY; p.font.italic = True
        elif any(k in ln for k in ['def ','if ','elif ','else:','return ','for ']):
            p.font.color.rgb = C_ACCENT
        else:
            p.font.color.rgb = C_PRIMARY

def cards2(sl, lt, li, rt, ri, y):
    w = Inches(4.2)
    for cx,t,items in [(Inches(0.6),lt,li),(Inches(5.2),rt,ri)]:
        ht = Inches(0.45+0.30*len(items)+0.15)
        b = sl.shapes.add_textbox(cx,y,w,ht)
        b.fill.solid(); b.fill.fore_color.rgb = C_BOX_BG
        b.line.color.rgb = C_BORDER; b.line.width = Pt(1)
        tf = b.text_frame; tf.word_wrap = True
        tf.margin_left = Inches(0.15); tf.margin_top = Inches(0.1)
        p0 = tf.paragraphs[0]
        p0.text = t; p0.font.size = Pt(18); p0.font.name = F_S
        p0.font.bold = True; p0.font.color.rgb = C_ACCENT
        for it in items:
            p = tf.add_paragraph()
            p.text = "\u2022  "+it; p.font.size = Pt(14); p.font.name = F_N; p.font.color.rgb = C_BODY
    return y+ht+Inches(0.15)


# ════════════════════════════════════════
#   MOTIVATION (Slides 1-3)
# ════════════════════════════════════════

def S01_title(prs):
    sl = prs.slides.add_slide(prs.slide_layouts[6]); wb(sl)
    b = sl.shapes.add_textbox(Inches(0.8), Inches(2.0), Inches(8.4), Inches(1.4))
    tf = b.text_frame; tf.word_wrap = True
    p = tf.paragraphs[0]
    p.text = "BOK-GRPO"; p.font.size = Pt(52); p.font.name = F_S
    p.font.bold = True; p.font.color.rgb = C_PRIMARY; p.alignment = PP_ALIGN.CENTER
    p2 = tf.add_paragraph()
    p2.text = "Best-of-K Group Relative Policy Optimization"
    p2.font.size = Pt(22); p2.font.name = F_S; p2.font.color.rgb = C_ACCENT
    p2.alignment = PP_ALIGN.CENTER
    rl(sl, Inches(3.8), Inches(3.7), 2.4)
    b2 = sl.shapes.add_textbox(Inches(1.5), Inches(4.2), Inches(7), Inches(0.9))
    tf2 = b2.text_frame; tf2.word_wrap = True
    p3 = tf2.paragraphs[0]
    p3.text = "Interleaved Point-to-Count for VLM Object Counting"
    p3.font.size = Pt(18); p3.font.name = F_N; p3.font.color.rgb = C_SECONDARY
    p3.alignment = PP_ALIGN.CENTER
    p4 = tf2.add_paragraph()
    p4.text = "Collapsing pass@K capability into pass@1"
    p4.font.size = Pt(16); p4.font.name = F_N; p4.font.color.rgb = C_SECONDARY
    p4.alignment = PP_ALIGN.CENTER

def S02_gap_problem(prs):
    sl = prs.slides.add_slide(prs.slide_layouts[6]); wb(sl)
    y = tb(sl, "The pass@K Gap Problem")
    y = tbl(sl, ["Metric", "Value", "Implication"],
        [["pass@1", "79.2%", "Current single-attempt accuracy"],
         ["pass@32", "95.2%", "Latent capability with 32 attempts"],
         ["Gap", "16.0pp", "Untapped potential to recover"]],
        y, cw=[2.0,2.0,4.8], hl=True)
    y += Inches(0.05)
    cards2(sl,
        "Standard GRPO  \u2717",
        ["A = (r \u2212 \u03bc) / \u03c3  \u2192 vanishing gradients when \u03c3\u21920",
         "14/16 rollouts correct \u2192 no learning signal",
         "Cannot differentiate among high-reward paths"],
        "BOK-GRPO  \u2713",
        ["A = (softmax(z/\u03c4) \u2212 1/K) \u00d7 K",
         "Always separates best from rest, even when all good",
         "\u03c4 annealing: explore broadly \u2192 focus on single best"],
        y)

def S03_paradigm(prs):
    sl = prs.slides.add_slide(prs.slide_layouts[6]); wb(sl)
    y = tb(sl, "Interleaved Point-to-Count Paradigm")
    y = tbl(sl, ["Turn", "Input", "Model Output", "Environment"],
        [["1", "Original image + question", '<point>{x,y, count:1}</point>', "Draw red dot #1"],
         ["2", "Image + 1 red dot", '<point>{x,y, count:2}</point>', "Draw red dot #2"],
         ["\u22ee", "Image + (k\u22121) dots", '<point>{x,y, count:k}</point>', "Draw red dot #k"],
         ["N", "Image + (N\u22121) dots", "<answer>N</answer>", "\u2192 STOP"]],
        y, cw=[0.8,2.5,3.0,2.5], hl=True)
    y += Inches(0.05)
    ib(sl, "Why Interleaved? (vs Single-Turn All-at-Once)",
        "Humans count by pointing one-by-one. Sequential pointing creates explicit attention shift: "
        "each red dot serves as visual memory, preventing double-counting and attention drift.", y)
    y += Inches(0.85)
    y = tbl(sl, ["Aspect", "Single-Turn", "Interleaved (Ours)", "Why Better"],
        [["Attention", "Must track all at once", "One object per turn", "Reduces cognitive load"],
         ["Feedback", "No mid-process signal", "Red dot = visual memory", "Prevents re-counting"],
         ["Error mode", "Catastrophic all-or-nothing", "Graceful per-step error", "Error isolation"]],
        y, cw=[1.5,2.1,2.4,2.8])

def S04_core_formula(prs):
    sl = prs.slides.add_slide(prs.slide_layouts[6]); wb(sl)
    y = tb(sl, "Core Algorithm: Softmax Advantage")
    y = sec(sl, "Standard GRPO \u2192 Fails on Easy Groups", y)
    y = fm(sl, "A\u1d62 = (r\u1d62 \u2212 \u03bc\u209c) / \u03c3\u209c        \u27f9   \u03c3\u209c \u2192 0 when 14/16 correct", y)
    y += Inches(0.1)
    y = sec(sl, "BOK-GRPO: Softmax-Temperature Advantage", y)
    y = fm(sl, "w\u1d62 = softmax( z\u1d62 / \u03c4 )        A\u1d62 = (w\u1d62 \u2212 1/K) \u00d7 K        where z\u1d62 = (r\u1d62 \u2212 \u03bc) / \u03c3", y)
    y += Inches(0.05)
    y = tbl(sl, ["\u03c4 Value", "Behavior", "Why This Range"],
        [["\u03c4 \u2192 0", "Winner-take-all (pure BoK)", "Maximum pressure, potentially unstable"],
         ["\u03c4 = 0.3\u20130.7", "Top-K concentration", "Practical: strong signal, still stable"],
         ["\u03c4 \u2192 \u221e", "Uniform (\u2248 std GRPO)", "No selection pressure"]],
        y, cw=[2.0,3.0,3.8])
    y += Inches(0.05)
    y = sec(sl, "Cosine Annealing: Why Start at \u03c4=0.5?", y)
    y = fm(sl, "\u03c4(t) = \u03c4_f + (\u03c4_0 \u2212 \u03c4_f) \u00d7 \u00bd \u00d7 (1 + cos(\u03c0t/T))        [0.5 \u2192 0.3]", y)
    ib(sl, "Why \u03c4_init=0.5, \u03c4_final=0.3?",
        "Early training: policy is noisy, broader credit distribution (\u03c4=0.5) avoids premature convergence. "
        "Late training: policy is refined, sharper selection (\u03c4=0.3) focuses on the single best path.", y)


# ════════════════════════════════════════
#   METHOD (Slides 5-8)
# ════════════════════════════════════════

def S05_routing(prs):
    sl = prs.slides.add_slide(prs.slide_layouts[6]); wb(sl)
    y = tb(sl, "4-Way Adaptive Routing")
    y = tbl(sl, ["Route", "Condition", "Advantage", "Why This Route"],
        [["1. LowVar", "\u03c3\u209c \u2264 \u03b5", "r \u2212 batch_mean", "Uniform group: use batch baseline"],
         ["2. AllCorrect", "All K correct", "A = 0", "Nothing to learn: skip"],
         ["3. EasyDrGRPO", "pass > \u03b8=0.50", "clip(z, \u00b12.5)", "Easy: z-norm suffices"],
         ["4. BoK Softmax", "Default", "(softmax(z/\u03c4)\u22121/K)\u00d7K", "Hard: need BoK selection"]],
        y, cw=[1.5,2.0,2.8,2.5], hl=True)
    y += Inches(0.05)
    ib(sl, "Why \u03b8_easy = 0.50?",
        "Empirically validated: below 0.50 too many groups incorrectly routed to EasyDrGRPO; "
        "above 0.50 too few groups get efficient z-norm gradients. "
        "0.50 gives 84.7% effective gradient coverage.", y)
    y += Inches(0.85)
    y = sec(sl, "Why EasyDrGRPO Filter?", y)
    y = bul(sl, "\u2022  When >50% rollouts succeed, z-norm gives valid gradients (enough variance)", y, sz=Pt(15))
    y = bul(sl, "\u2022  BoK softmax on easy groups wastes computation \u2014 z-norm is cheaper & sufficient", y, sz=Pt(15))
    y = bul(sl, "\u2022  Reserves BoK's concentrated selection for genuinely hard groups where it matters", y, sz=Pt(15))
    y += Inches(0.1)
    y = tbl(sl, ["LowVar", "AllCorrect", "EasyDrGRPO", "BoK", "Effective \u2265"],
        [["7%", "15%", "47%", "31%", "84.7%"]],
        y, cw=[1.76,1.76,1.76,1.76,1.76])

def S06_safety(prs):
    sl = prs.slides.add_slide(prs.slide_layouts[6]); wb(sl)
    y = tb(sl, "Safety & Stability Mechanisms")
    y = tbl(sl, ["Mechanism", "Implementation", "Why Needed"],
        [["NaN Guard", "nan_to_num(scores, nan=0)", "Trajectory failures produce NaN rewards"],
         ["Logit Cap", "clamp(logits, \u00b112.0)", "Extreme z \u2192 softmax overflow \u2192 NaN"],
         ["\u03c4-Adaptive Floor", "\u03c4_eff = max(\u03c4, |z_max|/C)", "Prevents over-sharp softmax"],
         ["Advantage Clip", "clamp(A, \u00b14.0)", "Bounds gradient magnitude"],
         ["Collapse Detection", "max(w)>0.9 \u2192 mix uniform", "Single-traj domination \u2192 instability"],
         ["KL Penalty", "\u03b2=0.03 \u00b7 D_KL(\u03c0\u2016\u03c0_ref)", "Prevent policy drift from reference"],
         ["PPO Clip", "clip(r(\u03b8), 1\u2212\u03b5, 1+0.28)", "Trust region constraint"]],
        y, cw=[2.0,3.0,3.8])
    y += Inches(0.05)
    ib(sl, "eff_intensity = LR \u00d7 clip_high \u00d7 ppo_epochs",
        "Sweet spot: 4\u20136e\u207b\u2077. V12 uses 1e-6\u00d70.28\u00d72 = 5.6e-7. "
        "Too high (>1e-6) \u2192 NaN. Too low (<3e-7) \u2192 no learning.", y)

def S07_mask_reward(prs):
    sl = prs.slides.add_slide(prs.slide_layouts[6]); wb(sl)
    y = tb(sl, "Dense Mask Point Reward")
    y = sec(sl, "Per-Point Reward (3 Outcomes)", y)
    y = tbl(sl, ["Outcome", "Condition", "Reward", "Rationale"],
        [["HIT (new)", "Point in unused mask", "1.0", "Correct identification"],
         ["HIT (dup)", "Point in used mask", "0.0", "Penalize re-counting"],
         ["MISS", "Point outside all masks", "decay(d)", "Distance-based shaping"]],
        y, cw=[1.5,2.5,1.5,3.3])
    y += Inches(0.05)
    y = sec(sl, "Distance Decay: Why 2-Phase Exponential?", y)
    y = tbl(sl, ["Phase", "Formula", "Why This Design"],
        [["Near (d \u2264 0.02)", "exp(\u2212\u03b1\u00b7d), \u03b1=20", "Smooth gradient toward correct location"],
         ["Far (d > 0.02)", "exp(\u2212\u03b1\u00b7d_f)\u00b7exp(\u221250(d\u2212d_f))", "Steep drop: clearly wrong, minimal reward"]],
        y, cw=[2.0,3.5,3.3])
    y += Inches(0.05)
    ib(sl, "Why Masks Instead of GT Points?",
        "RL training has NO per-step GT point annotations \u2014 only image + GT count. "
        "Pre-computed segmentation masks enable dense per-step reward without manual labeling. "
        "Mask-based matching handles objects of varying size (hit = inside contour, not just proximity).", y)

def S08_trajectory_reward(prs):
    sl = prs.slides.add_slide(prs.slide_layouts[6]); wb(sl)
    y = tb(sl, "Trajectory Reward Composition")
    y = fm(sl, "R = 0.6 \u00d7 R_answer  +  0.3 \u00d7 R_point  +  0.1 \u00d7 R_format", y)
    y += Inches(0.05)
    y = tbl(sl, ["Component", "Weight", "Design", "Why This Weight"],
        [["R_answer", "0.6", "Soft decay, cap=0.4", "Primary goal is correct count"],
         ["R_point", "0.3", "Dense mask per-step", "Prevents reward sparsity"],
         ["R_format", "0.1", "Binary compliance", "Ensures parseable output"]],
        y, cw=[1.5,1.0,2.5,3.8])
    y += Inches(0.05)
    y = sec(sl, "Answer Decay: Why Asymmetric Penalty?", y)
    y = tbl(sl, ["Direction", "k Factor", "Why Asymmetric"],
        [["Over-count (pred > GT)", "k = 2.0", "Hallucination is worse \u2014 counts non-existent objects"],
         ["Under-count (pred < GT)", "k = 1.0\u20131.5", "Missing objects is less severe, especially for large GT"]],
        y, cw=[2.5,2.0,4.3])
    y += Inches(0.05)
    ib(sl, "Why 0.6/0.3/0.1 and Not 1.0/0/0?",
        "Pure answer reward is too sparse for multi-turn RL (one signal per trajectory). "
        "Dense point reward (0.3) provides per-step learning signal. "
        "Format (0.1) prevents structural degeneration.", y)
    y += Inches(0.85)
    y = sec(sl, "Consistency Check", y)
    y = fm(sl, "If pred_answer \u2260 #pred_points:    R_answer \u00d7= 0.5", y)


# ════════════════════════════════════════
#   EXPERIMENTS (Slides 9-11)
# ════════════════════════════════════════

def S09_config(prs):
    sl = prs.slides.add_slide(prs.slide_layouts[6]); wb(sl)
    y = tb(sl, "Training Configuration")
    y = tbl(sl, ["Parameter", "Value", "Why This Choice"],
        [["Base Model", "Qwen2.5-VL-7B", "Strong VLM with point/grounding capability"],
         ["SFT Pre-train", "30k / 3537 steps", "Establish counting format before RL"],
         ["K (rollouts)", "16", "Enough diversity for softmax advantage"],
         ["Learning Rate", "1e\u207b\u2076", "Part of eff_intensity sweet spot"],
         ["PPO Epochs", "2", "Balance between update strength and stability"],
         ["Clip Ratio", "0.28", "eff_intensity = LR\u00d7clip\u00d7ppo = 5.6e-7"],
         ["KL Coeff", "0.03", "Prevent drift while allowing adaptation"],
         ["\u03c4 Schedule", "0.5 \u2192 0.3 cosine", "Explore-then-exploit temperature"],
         ["BOK_CLIP", "4.0", "Bound advantage magnitude"],
         ["\u03b8_easy", "0.50", "Validated optimal routing threshold"],
         ["Max Turns", "11", "Cover GT \u226410 with margin"],
         ["Reward", "0.6/0.3/0.1", "Answer + dense point + format"]],
        y, cw=[2.0,2.3,4.5])

def S10_data(prs):
    """Why mixed data."""
    sl = prs.slides.add_slide(prs.slide_layouts[6]); wb(sl)
    y = tb(sl, "Data Strategy")
    y = sec(sl, "Dataset Comparison", y)
    y = tbl(sl, ["Dataset", "Samples", "Easy/Med/Hard", "Use Case"],
        [["0_10 (sparse only)", "11,455", "30/41/29%", "V7-V12: sparse counting"],
         ["hard_only", "1,808", "18/40/42%", "V14: hard data focus"],
         ["mixed (hard+easy)", "5,792", "17/32/51%", "V14-16: balanced training"]],
        y, cw=[2.2,1.5,2.3,2.8])
    y += Inches(0.05)
    ib(sl, "Why Mixed Data (Not Pure Hard)?",
        "Pure hard data (V14-hard) showed: answer reward=0.74 but pixmo-test=79.4% (worse than V12). "
        "Easy examples provide gradient signal when hard examples are all-wrong. "
        "Mixed data maintains BoK routing diversity and prevents policy collapse.", y)
    y += Inches(0.85)
    y = sec(sl, "Impact of Data Choice", y)
    y = tbl(sl, ["Experiment", "Data", "pixmo-test", "Finding"],
        [["V12 (best)", "0_10 sparse", "82.04%", "Best overall accuracy"],
         ["V14-hard", "hard_only", "79.40%", "Too narrow, overfits to hard"],
         ["V14-mixed", "mixed", "80.34%", "Better than hard-only"],
         ["V15", "mixed + low LR", "80.15%", "LR too conservative"]],
        y, cw=[2.0,2.0,2.0,2.8], hl=False)

def S11_results(prs):
    sl = prs.slides.add_slide(prs.slide_layouts[6]); wb(sl)
    y = tb(sl, "Main Results")
    y = sec(sl, "Benchmark Comparison (GT 2\u201310)", y)
    y = tbl(sl, ["Model", "pixmo-test", "countbench", "Method"],
        [["SFT Base", "75.6%", "\u2014", "Supervised Only"],
         ["V7 Std GRPO", "80.72%", "\u2014", "Standard GRPO"],
         ["V14 Mixed", "80.34%", "78.21%", "Mixed + BoK"],
         ["V15 Low-LR", "80.15%", "79.02%", "Conservative"],
         ["V12 BOK-GRPO", "82.04%", "79.43%", "BOK-GRPO (best)"]],
        y, cw=[2.3,2.0,2.0,2.5], hl=True)
    y += Inches(0.05)
    y = sec(sl, "V12 Detailed Breakdown", y)
    y = tbl(sl, ["Split", "Samples", "Accuracy"],
        [["Overall", "529", "82.04%"],
         ["Easy (GT 0\u20135)", "239", "84.10%"],
         ["Hard (GT 5+)", "290", "80.34%"],
         ["With History", "\u2014", "83.36%"]],
        y, cw=[3.0,2.9,2.9])
    y += Inches(0.05)
    ib(sl, "Key Result",
        "+6.44pp over SFT (75.6\u219282.04%) | +1.32pp over Standard GRPO (80.72\u219282.04%) | "
        "pass@1\u2013@32 gap: 19.6pp \u2192 ~13pp (\u221233%)", y)

def S12_ablation(prs):
    sl = prs.slides.add_slide(prs.slide_layouts[6]); wb(sl)
    y = tb(sl, "Ablation: What Matters")
    y = tbl(sl, ["Design Choice", "Without", "With (Ours)", "Why It Helps"],
        [["Interleaved multi-turn", "All-at-once", "Sequential+feedback", "Visual memory via red dots"],
         ["Dense point reward", "Answer-only", "+mask point (0.3)", "Per-step signal prevents sparsity"],
         ["BoK softmax adv.", "Z-norm GRPO", "Softmax weighting", "+1.32pp: learns from best paths"],
         ["Mask matching", "Distance-only", "Hit/miss/duplicate", "Size-aware spatial grounding"],
         ["Consistency check", "None", "R_ans\u00d70.5 if \u2260", "Penalizes contradictory outputs"],
         ["Soft answer decay", "Binary 0/1", "exp decay, cap=0.4", "Finer gradient for wrong answers"],
         ["\u03c4 cosine annealing", "Fixed \u03c4=0.3", "0.5\u21920.3 cosine", "Explore early, exploit late"]],
        y, cw=[2.2,2.0,2.3,2.3])
    y += Inches(0.05)
    ib(sl, "Most Impactful",
        "Dense point reward (prevents reward hacking) and BoK softmax advantage (+1.32pp) "
        "are the two most critical design decisions.", y)


# ════════════════════════════════════════
#   FINDINGS (Slides 13-14)
# ════════════════════════════════════════

def S13_findings(prs):
    sl = prs.slides.add_slide(prs.slide_layouts[6]); wb(sl)
    y = tb(sl, "Key Findings")
    y = tbl(sl, ["Finding", "Evidence", "Implication"],
        [["+6.44pp over SFT", "75.6% \u2192 82.04%", "RL training is effective for counting"],
         ["+1.32pp over Std GRPO", "80.72% \u2192 82.04%", "BoK advantage > z-norm advantage"],
         ["pass@1\u2013@32 gap \u221233%", "19.6pp \u2192 ~13pp", "Capability collapse is working"],
         ["Zero NaN events", "eff_intensity=5.6e-7", "Safety mechanisms are sufficient"],
         ["96.6% format compliance", "Maintained throughout", "Format reward preserves structure"],
         ["Routing is data-driven", "Easy:47% BoK:31%", "Threshold \u03b8=0.50 is robust"]],
        y, cw=[2.5,2.8,3.5])
    y += Inches(0.05)
    y = sec(sl, "Stability Metrics", y)
    y = tbl(sl, ["Metric", "Value", "Status"],
        [["eff_intensity", "5.6e\u207b\u2077", "\u2713 In sweet spot (4\u20136e-7)"],
         ["NaN Events", "0", "\u2713 Fully stable"],
         ["Format Compliance", "96.6%", "\u2713 High"]],
        y, cw=[3.0,2.9,2.9])

def S14_future(prs):
    sl = prs.slides.add_slide(prs.slide_layouts[6]); wb(sl)
    y = tb(sl, "Future Directions")
    y = sec(sl, "Performance Targets", y)
    y = tbl(sl, ["Scenario", "Current", "Target", "Approach"],
        [["Sparse (\u226410 obj)", "82%", "\u226595%", "BOK-GRPO + curriculum"],
         ["Dense (11\u201350 obj)", "~30%", "\u226550%", "Dense scene + adaptive routing"]],
        y, cw=[2.5,1.5,1.5,3.3])
    y += Inches(0.1)
    cards2(sl,
        "Algorithm Extensions",
        ["Per-turn process reward (step-level advantage)",
         "Adaptive routing thresholds (dynamic \u03b8)",
         "History-efficient prompting",
         "Curriculum: easy \u2192 hard scheduling"],
        "Scaling & Transfer",
        ["Qwen2.5-VL-72B with BOK-GRPO",
         "Cross-domain: detection, segmentation",
         "Real-time streaming visual feedback",
         "Multi-category object counting"],
        y)

def S15_pseudocode(prs):
    sl = prs.slides.add_slide(prs.slide_layouts[6]); wb(sl)
    y = tb(sl, "Appendix: Algorithm Pseudocode")
    cb(sl, """
def BOK_GRPO(rollouts, K=16, \u03c4_init=0.5, \u03c4_final=0.3, T=90):
    scores = trajectory_rewards(rollouts)     # R = 0.6*ans + 0.3*pt + 0.1*fmt
    \u03c4 = \u03c4_f + (\u03c4_0 \u2212 \u03c4_f) * 0.5 * (1 + cos(\u03c0*t/T))    # cosine annealing

    for group in rollouts.group_by_prompt():
        \u03c3, pass_rate = group.std(), fraction(score > \u03b8)

        if \u03c3 \u2264 \u03b5:                              # Route 1: LowVar
            A = score \u2212 batch_mean
        elif all(score > \u03b8):                      # Route 2: AllCorrect
            A = 0
        elif pass_rate > \u03b8_easy:                  # Route 3: EasyDrGRPO
            A = clip((score \u2212 \u03bc)/\u03c3, \u00b12.5)
        else:                                     # Route 4: BoK Softmax
            z = (score \u2212 \u03bc) / \u03c3
            \u03c4_eff = max(\u03c4, max|z|/C, \u03c4_min)         # adaptive floor
            w = softmax(z / \u03c4_eff)
            w = (1\u2212\u03b1)*w + \u03b1/K                      # uniform mixing
            A = (w \u2212 1/K) * K                       # BoK advantage

        A = clamp(A, \u2212BOK_CLIP, BOK_CLIP)          # final clip

    # PPO update
    L = \u2212E[min(r(\u03b8)*A, clip(r(\u03b8), 1\u00b1\u03b5)*A)] + \u03b2*D_KL(\u03c0_\u03b8 || \u03c0_ref)
""", y, h=Inches(5.6), fs=Pt(13))


def main():
    prs = Presentation()
    prs.slide_width = Inches(10)
    prs.slide_height = Inches(7.5)

    # Motivation
    S01_title(prs)
    S02_gap_problem(prs)
    S03_paradigm(prs)
    S04_core_formula(prs)
    # Method
    S05_routing(prs)
    S06_safety(prs)
    S07_mask_reward(prs)
    S08_trajectory_reward(prs)
    # Experiments
    S09_config(prs)
    S10_data(prs)
    S11_results(prs)
    S12_ablation(prs)
    # Findings
    S13_findings(prs)
    S14_future(prs)
    # Appendix
    S15_pseudocode(prs)

    out = '/mnt/shared-storage-user/zhangchenhao/work/EasyR1-latest/docs/BOK_GRPO_Presentation.pptx'
    prs.save(out)
    print(f"Saved: {out}")
    print(f"Slides: {len(prs.slides)}")

if __name__ == '__main__':
    main()
