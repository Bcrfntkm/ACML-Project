#!/usr/bin/env python3
"""
Manual gold annotation: 25 new paragraphs from 10 different papers.
Each paragraph annotated by reading text and identifying chemistry entities.

Combined with gold_chem0021.jsonl → 30 gold samples total.

Saves:
  data/training/gold_test/gold_full.jsonl      — all 30 paragraphs
  data/training/gold_test/gold_full_llama.jsonl — LLaMA's BIO for same paragraphs
  data/training/gold_test/labels.json
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

_SCIENCE = Path(__file__).resolve().parent.parent

# ── BIO helpers (same as convert_reactions_to_bio.py) ─────────────────────────

def _tokenize(text: str):
    return list(re.finditer(r'\S+', text))

def _find_spans(text: str, value: str) -> list[tuple[int, int]]:
    spans, start = [], 0
    while True:
        i = text.find(value, start)
        if i == -1:
            break
        spans.append((i, i + len(value)))
        start = i + 1
    return spans

def _find_nth(text: str, value: str, n: int) -> tuple[int, int] | None:
    """Return the n-th (0-indexed) occurrence of value in text."""
    start = 0
    for _ in range(n + 1):
        i = text.lower().find(value.lower(), start)
        if i == -1:
            return None
        pos = (i, i + len(value))
        start = i + 1
    return pos

def _bio_labels(text: str, annotations: list[dict]) -> list[str]:
    matches = _tokenize(text)
    labels = ["O"] * len(matches)
    for ann in sorted(annotations, key=lambda a: a["start"]):
        s, e, lbl = ann["start"], ann["end"], ann["label"]
        first = True
        for i, m in enumerate(matches):
            if m.end() <= s or m.start() >= e:
                continue
            labels[i] = f"B-{lbl}" if first else f"I-{lbl}"
            first = False
    return labels

def build_sample(text: str, entities: list[tuple], source: str) -> dict:
    """
    entities: list of (value, label) or (value, label, occurrence_index)
    occurrence_index defaults to 0 (first match).
    """
    annotations = []
    used_spans: set[tuple[int, int]] = set()
    for item in entities:
        value, label = item[0], item[1]
        occ = item[2] if len(item) > 2 else 0
        span = _find_nth(text, value, occ)
        if span is None:
            print(f"  [WARN] Not found: {value!r} (occ={occ}) in:\n    {text[:80]}")
            continue
        if span in used_spans:
            continue
        used_spans.add(span)
        annotations.append({"start": span[0], "end": span[1], "label": label})

    tokens = [m.group() for m in _tokenize(text)]
    labels = _bio_labels(text, annotations)
    return {"tokens": tokens, "labels": labels, "source": source}


# ── LLaMA BIO from rm_*.json ──────────────────────────────────────────────────

_ROLE_MAP = {
    "product": "PRODUCT", "reactant": "REACTANT", "catalyst": "CATALYST",
    "solvent": "SOLVENT", "temperature": "TEMP", "time": "TIME",
    "yield": "YIELD", "othercondition": "COND", "other condition": "COND",
    "other": "COND", "atmosphere": "COND", "workup reagents": "COND",
    "workup": "COND", "reaction type": "COND", "procedure": "COND",
    "additive": "COND", "base": "COND", "acid": "COND", "ligand": "COND",
    "ph": "COND", "pressure": "COND", "vacuum condition": "COND",
}

def _norm_role(k: str) -> str:
    return _ROLE_MAP.get(k.strip().lower(), "COND")

def _find_spans_fuzzy(text: str, value: str, min_ratio: float = 0.75):
    val = value.strip()
    if not val:
        return []
    spans = _find_spans(text, val)
    if spans:
        return spans
    lo = text.lower().find(val.lower())
    if lo != -1:
        return [(lo, lo + len(val))]
    parts = [p.strip() for p in val.split(',') if len(p.strip()) > 3]
    if len(parts) > 1:
        found = []
        for part in parts:
            i = text.lower().find(part.lower())
            if i != -1:
                found.append((i, i + len(part)))
        if found:
            return found
    val_toks = set(re.findall(r'\w+', val.lower()))
    if len(val_toks) <= 1:
        return []
    words = list(re.finditer(r'\S+', text))
    if not words:
        return []
    win = max(1, min(len(val_toks) + 2, len(words)))
    best_r, best_span = 0.0, None
    for i in range(len(words) - win + 1):
        chunk = text[words[i].start():words[i + win - 1].end()].lower()
        ct = set(re.findall(r'\w+', chunk))
        if not ct:
            continue
        r = len(val_toks & ct) / len(val_toks | ct)
        if r > best_r:
            best_r, best_span = r, (words[i].start(), words[i + win - 1].end())
    if best_r >= min_ratio and best_span:
        return [best_span]
    return []

def llama_to_bio(text: str, reactions: list[dict]) -> dict:
    annotations = []
    seen: set[tuple[int, int, str]] = set()
    KEY = {"product", "reactant", "catalyst", "solvent"}
    for rx in reactions:
        grounded = any(
            isinstance(v, str) and len(v) > 3 and text.lower().find(v.strip().lower()) != -1
            for k, v in rx.items() if k.strip().lower() in KEY
        )
        if not grounded:
            continue
        for role, val in rx.items():
            if not isinstance(val, str) or not val.strip():
                continue
            lbl = _norm_role(role)
            for s, e in _find_spans_fuzzy(text, val.strip()):
                key = (s, e, lbl)
                if key not in seen:
                    seen.add(key)
                    annotations.append({"start": s, "end": e, "label": lbl})
    tokens = [m.group() for m in _tokenize(text)]
    labels = _bio_labels(text, annotations)
    return {"tokens": tokens, "labels": labels, "source": "llama"}


# ── Gold annotations (25 new paragraphs, hand-labeled) ───────────────────────
# Each entry: (text, entities, source_file, paragraph_reactions_for_llama)
# entities: list of (value, label) or (value, label, occurrence)

GOLD_NEW = [

    # ── molecules-24-00006: Cu coordination polymer synthesis ─────────────────
    (
        "Synthesis of [Cu(µ-cpna)(phen)(H2O)]n (1) A mixture of CuCl2·2H2O "
        "(34.1 mg, 0.2 mmol), H2cpna (51.8 mg, 0.2 mmol), phen (39.6 mg,0.2 mmol), "
        "NaOH (16.0 mg, 0.4 mmol), and H2O (10 mL) was stirred at room temperature "
        "for 15 min.",
        [
            ("[Cu(µ-cpna)(phen)(H2O)]n (1)", "PRODUCT"),
            ("CuCl2·2H2O", "REACTANT"),
            ("H2cpna", "REACTANT"),
            ("phen", "REACTANT", 1),        # 2nd occurrence (1st is inside product formula)
            ("NaOH", "REACTANT"),
            ("H2O", "SOLVENT", 1),          # 2nd H2O = the 10 mL solvent
            ("room temperature", "TEMP"),
            ("15 min", "TIME"),
        ],
        "rm_molecules-24-00006.json",
    ),

    # ── molecules-29-00091: Nitrile oxide chemistry with Et3N ────────────────
    (
        "Et3N (0.79 mL, 5.69 mmoles, 1.2 equiv.) was added dropwise, and the yellow "
        "solution was stirred for 15 min at 0 ◦C. A DCM solution (100 mL) of "
        "phenylhydroxamoyl chloride 14a (0.811 g, 5.21 mmoles, 1.1 equiv.) was then "
        "added via a pressure-equalising dropping funnel over a period of 1 h, while "
        "the temperature was maintained at 0 ◦C. After the addition, the reaction "
        "mixture was left stirring for another 1 h, and the reaction temperature was "
        "gradually allowed to reach 20 ◦C.",
        [
            ("Et3N", "REACTANT"),
            ("phenylhydroxamoyl chloride 14a", "REACTANT"),
            ("DCM", "SOLVENT"),
            ("0 ◦C", "TEMP"),
            ("15 min", "TIME"),
            ("1 h", "TIME"),
            ("0 ◦C", "TEMP", 1),
            ("1 h", "TIME", 1),
            ("20 ◦C", "TEMP"),
        ],
        "rm_molecules-29-00091.json",
    ),

    # ── RA-011-D1RA08258B: Mn coordination complex ───────────────────────────
    (
        "A solution of bpy (0.031 g, 0.2 mmol) in methanol (6 ml) was added to an "
        "aqueous solution of MnCl2 (0.025 g, 0.2 mmol) and H2pydco (0.03 g, 0.2 mmol) "
        "in water (6 ml) and the mixture was reuxed at 90 NoneC for 6 h. Aer 10 days, "
        "yellow crystals were obtained by slow evaporation from the reaction mixture at "
        "room temperature in 70% yield (60 mg) (based on Mn) (mp 180 NoneC).",
        [
            ("bpy", "REACTANT"),
            ("MnCl2", "REACTANT"),
            ("H2pydco", "REACTANT"),
            ("methanol", "SOLVENT"),
            ("water", "SOLVENT"),
            ("yellow crystals", "PRODUCT"),
            ("90 NoneC", "TEMP"),
            ("6 h", "TIME"),
            ("room temperature", "TEMP"),
            ("70%", "YIELD"),
        ],
        "rm_RA-011-D1RA08258B.json",
    ),

    # ── copper_acetate: CuOAc2-mediated C–S coupling ────────────────────────
    (
        "In a typical procedure, 1a (50 mg, 0.32 mmol) and Cu(OAc)2 "
        "(57.9 mg, 0.32 mmol) were taken in a 10 mL tube sealed with a Teflon-lined "
        "cap; and to it was added DMSO (0.25 mL).",
        [
            ("1a", "REACTANT"),
            ("Cu(OAc)2", "CATALYST"),    # copper acetate = oxidant/catalyst
            ("DMSO", "SOLVENT"),
        ],
        "rm_copper_acetate.json",
    ),

    # ── copper_acetate: Product 2b isolated in 89% yield ─────────────────────
    (
        "We were excited to find that, with 2.0 equiv of copper acetate and DMSO as "
        "the solvent, 2-(3-(methylthio)naphthalen-2-yl)pyridine (2b) was obtained as "
        "a single product in 89% yield at 125 °C (Table 1).",
        [
            ("copper acetate", "CATALYST"),
            ("DMSO", "SOLVENT"),
            ("2-(3-(methylthio)naphthalen-2-yl)pyridine (2b)", "PRODUCT"),
            ("89%", "YIELD"),
            ("125 °C", "TEMP"),
        ],
        "rm_copper_acetate.json",
    ),

    # ── copper_acetate: Indole + NaH setup ───────────────────────────────────
    (
        "Indole (1.0 g, 8.54 mmol) was placed in a 100 mL two-neck reaction flask, "
        "and flushed with nitrogen. DMF (25 mL) and NaH (819 mg, 34.1 mmol) were added.",
        [
            ("Indole", "REACTANT"),
            ("nitrogen", "COND"),
            ("DMF", "SOLVENT"),
            ("NaH", "REACTANT"),
        ],
        "rm_copper_acetate.json",
    ),

    # ── molecules-31-01513: Schiff base synthesis (BIMPB) ────────────────────
    (
        "Synthesis of (E)-3-(((1H-Benzo[d]imidazol-2-yl)methyl)imino)-1-phenylbutan-1-one "
        "(BIMPB)In this experiment, we used chemicals from Sigma-Aldrich (St. Louis, MO, USA) "
        "with-out analysis to synthesize (E)-3-(((1H-benzo[d]imidazol-2-yl)methyl)imino)-"
        "1-phenylbutan-1-one, and the following optimized procedure was followed:"
        "(1H-benzo[d]imidazol-2-yl)methanamine (1.0 mmol) and 1-phenylbutane-1,3-dione"
        "(1.0 mmol) were mixed in ethanol (EtOH) (20 mL) as the reaction solvent.",
        [
            ("(E)-3-(((1H-Benzo[d]imidazol-2-yl)methyl)imino)-1-phenylbutan-1-one (BIMPB)",
             "PRODUCT"),
            ("(1H-benzo[d]imidazol-2-yl)methanamine", "REACTANT", 1),  # 2nd occ = standalone reagent
            ("1-phenylbutane-1,3-dione", "REACTANT"),
            ("ethanol", "SOLVENT"),
        ],
        "rm_molecules-31-01513.json",
    ),

    # ── 41598_2026_Article_44568: PCC oxidation setup ────────────────────────
    (
        "Synthesis 5-Formylisophthalic acid (I)Pyridinium chlorochromate "
        "(2 equiv.; 7.92 mmol; 1.71 g) and neutral alumina (1 g/mmol of PCC; 8 g) "
        "were stirred under reflux in an argon atmosphere for 1 h in 17 mL of "
        "dichloromethane (DCM).",
        [
            ("5-Formylisophthalic acid (I)", "PRODUCT"),   # synthesis target from header
            ("Pyridinium chlorochromate", "REACTANT"),
            ("neutral alumina", "COND"),
            ("argon", "COND"),
            ("dichloromethane", "SOLVENT"),
            ("1 h", "TIME"),
            ("reflux", "TEMP"),
        ],
        "rm_41598_2026_Article_44568.json",
    ),

    # ── 41598_2026_Article_44568: diethyl isophthalate added ─────────────────
    (
        "After cooling to room temperature, a solution of commercially available "
        "diethyl 5-(hydroxymethyl)isophthalate (3.96 mmol; 1 g) in 13 mL of DCM "
        "was added.",
        [
            ("room temperature", "TEMP"),
            ("diethyl 5-(hydroxymethyl)isophthalate", "REACTANT"),
            ("DCM", "SOLVENT"),
        ],
        "rm_41598_2026_Article_44568.json",
    ),

    # ── 41598_2026_Article_44568: overall synthesis summary ──────────────────
    (
        "Starting from commercially available diethyl 5-(hydroxymethyl)isophthalate, "
        "oxidation with pyridinium chlorochromate (PCC) gave diethyl 5-formylisophthalate"
        "28,29, which was subsequently hydrolyzed to afford compound I in an overall yield "
        "ofI; 5-formylisophthalic acidDiethyl 5-(hydroxymethyl)isophtalate a. PCC (2 equiv), "
        "neutral alumina (1 g/mmol of PCC), DCM, reflux, 1 h, 98%; b. NaOH (10 equiv), "
        "MeOH, r. t., 6 h, 68%.",
        [
            ("diethyl 5-(hydroxymethyl)isophthalate", "REACTANT"),
            ("pyridinium chlorochromate (PCC)", "REACTANT"),
            ("diethyl 5-formylisophthalate", "PRODUCT"),
            ("DCM", "SOLVENT"),
            ("reflux", "TEMP"),
            ("1 h", "TIME"),
            ("98%", "YIELD"),
            ("NaOH", "REACTANT"),
            ("MeOH", "SOLVENT"),
            ("r. t.", "TEMP"),
            ("6 h", "TIME"),
            ("68%", "YIELD"),
        ],
        "rm_41598_2026_Article_44568.json",
    ),

    # ── crystals-14-00144: Cu/Zn L1 complex synthesis ───────────────────────
    (
        "General Procedure for the Synthesis of Copper and Zinc L1 Complexes "
        "Zn(NO3)2·6H2O or Cu(NO3)2·6H2O (4 equiv., 23.8/19.5 mg) and L1 "
        "(1 equiv., 10 mg)were dissolved in acetonitrile (1 mL) at room temperature.",
        [
            ("Zn(NO3)2·6H2O", "REACTANT"),
            ("Cu(NO3)2·6H2O", "REACTANT"),
            ("L1", "REACTANT"),
            ("acetonitrile", "SOLVENT"),
            ("room temperature", "TEMP"),
        ],
        "rm_crystals-14-00144-v3.json",
    ),

    # ── crystals-14-00144: Cu L2 complex ────────────────────────────────────
    (
        "General Procedure for the Synthesis of Copper L2 Complex "
        "Cu(NO3)2·6H2O (4 equiv., 19.3 mg) and L2 (1 equiv., 10 mg) were dissolved "
        "in 1-butanol (1 mL) at room temperature.",
        [
            ("Cu(NO3)2·6H2O", "REACTANT"),
            ("L2", "REACTANT"),
            ("1-butanol", "SOLVENT"),
            ("room temperature", "TEMP"),
        ],
        "rm_crystals-14-00144-v3.json",
    ),

    # ── molecules-27-06421: Mannich-type condensation with t-octylphenol ─────
    (
        "4-t-Octylphenol (268 g, 1.30 mol) was added and the mixture heated under "
        "reflux for 1 h. Toluene (333 mL) was added and the solvent removed under "
        "vacuum as the methanol/toluene azeotrope.",
        [
            ("4-t-Octylphenol", "REACTANT"),
            ("reflux", "TEMP"),
            ("1 h", "TIME"),
            ("Toluene", "SOLVENT"),
        ],
        "rm_molecules-27-06421.json",
    ),

    # ── molecules-27-06421: paraformaldehyde addition ─────────────────────────
    (
        "A slurry of paraformaldehyde (120 g, 4.0 mol) in toluene (200 mL) was "
        "added slowly with continuous distillation and heated for a further 2 h.",
        [
            ("paraformaldehyde", "REACTANT"),
            ("toluene", "SOLVENT"),
            ("2 h", "TIME"),
        ],
        "rm_molecules-27-06421.json",
    ),

    # ── molecules-29-03927: Sm heterobimetallic complex ──────────────────────
    (
        "The synthesis of 1Sm was performed as follows: Starting from [Sm(hfac)3] "
        "(0.224 g;0.29 mmol) and [Co(acac)3] (0.096 g; 0.27 mmol) in toluene (40 mL), "
        "the product [Sm(hfac)3 Co(acac)3] (0.226 g; 0.20 mmol; 74% yield) was recovered.",
        [
            ("[Sm(hfac)3]", "REACTANT"),
            ("[Co(acac)3]", "REACTANT"),
            ("toluene", "SOLVENT"),
            ("[Sm(hfac)3 Co(acac)3]", "PRODUCT"),
            ("74%", "YIELD"),
        ],
        "rm_molecules-29-03927.json",
    ),

    # ── molecules-29-03927: Dy heterobimetallic complex ──────────────────────
    (
        "The synthesis of 1Dy was performed as follows: Starting from [Dy(hfac)3] "
        "(0.106 g;0.14 mmol) and [Co(acac)3] (0.048 g; 0.13 mmol) in toluene (40 mL), "
        "the product [Dy(hfac)3 Co(acac)3] (0.097 g; 0.09 mmol; 63% yield) was recovered.",
        [
            ("[Dy(hfac)3]", "REACTANT"),
            ("[Co(acac)3]", "REACTANT"),
            ("toluene", "SOLVENT"),
            ("[Dy(hfac)3 Co(acac)3]", "PRODUCT"),
            ("63%", "YIELD"),
        ],
        "rm_molecules-29-03927.json",
    ),

    # ── molecules-29-03927: La-Ru heterobimetallic ───────────────────────────
    (
        "The synthesis of [La(hfac)3Ru(acac)3] was performed as follows: "
        "To a suspension of [La(hfac)3] (0.351 g; 0.46 mmol) in toluene (25 mL), "
        "[Ru(acac)3](0.184 g; 0.46 mmol) was added.",
        [
            ("[La(hfac)3Ru(acac)3]", "PRODUCT"),
            ("[La(hfac)3]", "REACTANT"),
            ("[Ru(acac)3]", "REACTANT"),
            ("toluene", "SOLVENT"),
        ],
        "rm_molecules-29-03927.json",
    ),

    # ── molecules-25-01573: Cu(II) quinoline piperidine complex ──────────────
    (
        "Synthesis of [Cu(quin)2(pipe)] (6) Copper(II) chloride dihydrate "
        "(80 mg, 0.47 mmol) was added to the solution of piperidine (1.5 mL)"
        "in acetonitrile (10 mL) in an Erlenmeyer flask.",
        [
            ("[Cu(quin)2(pipe)] (6)", "PRODUCT"),
            ("Copper(II) chloride dihydrate", "REACTANT"),
            ("piperidine", "REACTANT"),
            ("acetonitrile", "SOLVENT"),
        ],
        "rm_molecules-25-01573.json",
    ),

    # ── molecules-27-03218: Trinuclear Cu3 radical complex ───────────────────
    (
        "Synthesis of Complex Cu3(NIT2Ph)2Cl410 mL of a methanol solution of "
        "CuCl2 (64 mg, 0.48 mmol), which was previouslydried in a desiccator, "
        "was added to a 10 mL of a methanol solution of (NIT2PhOH) "
        "(100 mg, 0.24 mmol).",
        [
            ("Cu3(NIT2Ph)2", "PRODUCT"),
            ("CuCl2", "REACTANT"),
            ("(NIT2PhOH)", "REACTANT"),
            ("methanol", "SOLVENT"),
        ],
        "rm_molecules-27-03218.json",
    ),

    # ── chem0021-7179: copper(I) pyrrolidide via n-BuLi ──────────────────────
    (
        "Preparation of copper(I) pyrrolidide (3): A solution of pyrrolidine "
        "(1.11 mL, 13.50 mmol) in tetrahydrofuran (10 mL) was treated drop-wise "
        "with n-butyllithium in hexanes (5.20 mL, 13.00 mmol, 2.5 m) at room "
        "temperature to give a colourless solution.",
        [
            ("copper(I) pyrrolidide (3)", "PRODUCT"),
            ("pyrrolidine", "REACTANT"),
            ("tetrahydrofuran", "SOLVENT"),
            ("n-butyllithium in hexanes", "REACTANT"),
            ("room temperature", "TEMP"),
        ],
        "rm_chem0021-7179.json",
    ),

    # ── chem0021-7179: copper(I) dicyclohexylamide at 0°C ────────────────────
    (
        "Preparation of copper(I) dicyclohexylamide (1): A solution of "
        "di-cyclohexylamine (0.80 mL, 4.00 mmol) in tetrahydrofuran (10 mL) "
        "was treated dropwise with n-butyllithium in hexanes (2.50 mL, 4.00 mmol, "
        "1.6 m) at 0 8C.",
        [
            ("copper(I) dicyclohexylamide (1)", "PRODUCT"),
            ("di-cyclohexylamine", "REACTANT"),
            ("tetrahydrofuran", "SOLVENT"),
            ("n-butyllithium in hexanes", "REACTANT"),
            ("0 8C", "TEMP"),                # garbled "0°C" from PDF parser
        ],
        "rm_chem0021-7179.json",
    ),

    # ── chem0021-7179: 1-pyrroline side-product from thermal decomposition ───
    (
        "How-ever, 3 was also sensitive to thermal decomposition at room temperature "
        "or above, to give the b-hydride elimination prod-uct, 1-pyrroline, as a "
        "side-product in 10 % yield in [D6]benzene and 6 % yield in [D6]DMSO.",
        [
            ("1-pyrroline", "PRODUCT"),
            ("room temperature", "TEMP"),
            ("10 %", "YIELD"),
            ("[D6]benzene", "SOLVENT"),
            ("6 %", "YIELD"),
            ("[D6]DMSO", "SOLVENT"),
        ],
        "rm_chem0021-7179.json",
    ),

    # ── ic5c01578: ammonium hydroxide addition ───────────────────────────────
    (
        "After stirring the heterogeneous mixture for 3 min, concentrated (29% w/v) "
        "aqueous ammonium hydroxide (0.36 mL, 5.52 mmol, 2.2 equiv) was added, and "
        "the mixture darkened in color.",
        [
            ("ammonium hydroxide", "REACTANT"),
            ("3 min", "TIME"),
        ],
        "rm_ic5c01578.json",
    ),

    # ── crystals-13-01669-v2: Cu2 trinuclear complex ─────────────────────────
    (
        "Synthesis of [Cu2(cpida)(H2O)4][Cu(cpida)]·3H2O (1)H3cpida "
        "(0.167 g, 0.66 mmol) was dissolved in distilled water (12 mL) and "
        "coppersulfate (0.159 g, 1.00 mmol) was added.",
        [
            ("[Cu2(cpida)(H2O)4][Cu(cpida)]·3H2O", "PRODUCT"),
            ("H3cpida", "REACTANT", 1),   # 2nd occurrence (1st is inside product formula)
            ("distilled water", "SOLVENT"),
            ("coppersulfate", "REACTANT"),
        ],
        "rm_crystals-13-01669-v2.json",
    ),

    # ── molecules-29-03903: morpholine acylation ─────────────────────────────
    (
        "A suspension of morpholine (8.70 g, 0.100 mol), sodium carbonate "
        "(10.60 g, 0.100 mol), and 10 mL of CH2Cl2 was stirred at 0 ◦C. "
        "2-Oxopropanoyl chloride was added dropwise to the obtained cooled "
        "suspension and the mixture was stirred at room temperature for 1 h. "
        "After filtration, the mixture was washed with water (3 × 100 cm3) "
        "and dried in the air to afford the crude product as mobile yellow oil.",
        [
            ("morpholine", "REACTANT"),
            ("sodium carbonate", "REACTANT"),
            ("CH2Cl2", "SOLVENT"),
            ("0 ◦C", "TEMP"),
            ("2-Oxopropanoyl chloride", "REACTANT"),
            ("room temperature", "TEMP"),
            ("1 h", "TIME"),
        ],
        "rm_molecules-29-03903.json",
    ),

    # ── RA-011-D1RA08258B: complex 11 synthesis (different segment) ───────────
    (
        "A solution of bpy (0.031 g, 0.2 mmol) in methanol (6 ml) was added to an "
        "aqueous solution of CuCl2 (0.027 g, 0.2 mmol) and H2pydco (0.03 g, 0.2 mmol) "
        "in water (6 ml) and the mixture was refluxed at 90 °C for 6 h to give "
        "blue crystals in 65% yield.",
        [
            ("bpy", "REACTANT"),
            ("CuCl2", "REACTANT"),
            ("H2pydco", "REACTANT"),
            ("methanol", "SOLVENT"),
            ("water", "SOLVENT"),
            ("90 °C", "TEMP"),
            ("6 h", "TIME"),
            ("blue crystals", "PRODUCT"),
            ("65%", "YIELD"),
        ],
        "rm_RA-011-D1RA08258B.json",
    ),
]


# ── Load existing gold_chem0021 ───────────────────────────────────────────────

def load_existing_gold() -> list[dict]:
    p = _SCIENCE / "data/training/gold_test/gold_chem0021.jsonl"
    if not p.exists():
        return []
    return [json.loads(l) for l in p.read_text().splitlines() if l.strip()]


# ── Load LLaMA reactions for each paragraph ───────────────────────────────────

def load_llama_reactions(source_file: str, text: str) -> list[dict]:
    """Find the paragraph in source annotation file and return its reactions."""
    path = _SCIENCE / "data/annotations" / source_file
    if not path.exists():
        return []
    try:
        d = json.loads(path.read_text())
        for p in d.get("paragraphs", []):
            if p.get("text", "").strip() == text.strip():
                return p.get("reactions", [])
        # Fallback: partial match (first 80 chars)
        for p in d.get("paragraphs", []):
            if p.get("text", "").strip()[:80] == text.strip()[:80]:
                return p.get("reactions", [])
    except Exception:
        pass
    return []


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    out_dir = _SCIENCE / "data/training/gold_test"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load existing 5 gold samples
    existing = load_existing_gold()
    print(f"[INFO] Loaded {len(existing)} existing gold samples (gold_chem0021)")

    # Build new 25 gold samples
    new_gold = []
    new_llama = []
    for i, entry in enumerate(GOLD_NEW):
        text, entities, src = entry
        sample = build_sample(text, entities, f"gold_human:{src}")
        new_gold.append(sample)

        # LLaMA BIO for the same paragraph
        reactions = load_llama_reactions(src, text)
        llama_sample = llama_to_bio(text, reactions)
        llama_sample["source"] = f"llama:{src}"
        new_llama.append(llama_sample)

        non_o = sum(1 for l in sample["labels"] if l != "O")
        llama_non_o = sum(1 for l in llama_sample["labels"] if l != "O")
        print(f"  [{i+1:2d}] gold={non_o} non-O tokens  |  llama={llama_non_o} non-O tokens  | {text[:60]}...")

    # Combine all 30 gold samples
    all_gold = existing + new_gold
    all_llama = new_llama  # LLaMA for new 25 only (don't have reactions for existing 5)

    # Also create LLaMA BIO for the 5 existing paragraphs (from rm_chem0021-7179)
    # by loading from the annotation file
    chem_file = "rm_chem0021-7179.json"
    existing_texts = [
        "Isolation of the intermediate in the synthesis of copper(I) piperi-dide (6): A solution of copper(I) mesityl (365 mg, 2.00 mmol) in tet-rahydrofuran (2 mL) was treated with piperidine (988 mL, 10.00 mmol) and stirred at room temperature for 5 min after which a yellow precipitate was present.",
        "Preparation of copper(I) piperidide (4): A solution of copper(I) mesityl (1.462 g, 8.00 mmol) in tetrahydrofuran (10 mL) was treated with piperidine (3.95 mL, 40.00 mmol) at room temperature.",
        "In [D6]DMSO, the reaction of copper(I) amide 4 with iodobenzene gave the arylamine product in 87 % yield with some piperidine side-product also formed (8 % yield).",
        "After stirring for 10 min, the solution was transferred dropwise to a suspension of copper(I) chloride (1.39 g, 14.00 mmol) in tetrahydrofuran (50 mL) at room temperature.",
        "Preparation of copper(I) benzylamide (5): A solution of copper(I) mesityl (439 mg, 2.40 mmol) in tetrahydrofuran (1 mL) was treated with benzylamine (288 mL, 2.64 mmol) at room temperature.",
    ]
    existing_llama = []
    for text in existing_texts:
        reactions = load_llama_reactions(chem_file, text)
        ls = llama_to_bio(text, reactions)
        ls["source"] = f"llama:{chem_file}"
        existing_llama.append(ls)

    all_llama_full = existing_llama + all_llama  # 30 total

    # ── Save ──────────────────────────────────────────────────────────────────
    gold_out = out_dir / "gold_full.jsonl"
    with gold_out.open("w") as f:
        for s in all_gold:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")

    llama_out = out_dir / "gold_full_llama.jsonl"
    with llama_out.open("w") as f:
        for s in all_llama_full:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")

    # Labels
    all_labels = {l for s in all_gold for l in s["labels"]}
    entity_labels = sorted({l[2:] for l in all_labels if l != "O"})
    label_list = ["O"] + [f"B-{l}" for l in entity_labels] + [f"I-{l}" for l in entity_labels]
    (out_dir / "labels.json").write_text(json.dumps(label_list, indent=2))

    print(f"\n[INFO] Gold: {len(all_gold)} samples → {gold_out}")
    print(f"[INFO] LLaMA: {len(all_llama_full)} samples → {llama_out}")
    print(f"[INFO] Labels: {label_list}")

    # Summary stats
    from collections import Counter
    gold_counts = Counter(l for s in all_gold for l in s["labels"] if l != "O")
    llama_counts = Counter(l for s in all_llama_full for l in s["labels"] if l != "O")
    print(f"\n{'Label':<14} {'Gold':>6} {'LLaMA':>7}")
    print("-" * 28)
    for lbl in entity_labels:
        gb = gold_counts.get(f"B-{lbl}", 0)
        lb = llama_counts.get(f"B-{lbl}", 0)
        print(f"B-{lbl:<12} {gb:>6} {lb:>7}")


if __name__ == "__main__":
    main()
