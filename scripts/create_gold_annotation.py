#!/usr/bin/env python3
"""
Manual gold annotation for 5 synthesis paragraphs from chem0021-7179
(copper(I) amide chemistry paper).

Each paragraph was tokenized with the same _tokenize() function used in
convert_reactions_to_bio.py, then labeled token-by-token by a human annotator
reading the original text.

Saves to: data/training/gold_test/gold_chem0021.jsonl + labels.json
"""
from __future__ import annotations
import json
import re
from pathlib import Path

# ── same tokenizer as in convert_reactions_to_bio.py ──────────────────────────
def _tokenize(text: str):
    return [m.group() for m in re.finditer(r'\S+', text)]

# ── Gold-annotated samples ────────────────────────────────────────────────────
# Format: (text, {token_index: label})
# Only non-O labels listed; all others are O.
# B- = beginning of entity, I- = inside (continuation)

GOLD_SAMPLES = [

    # Para 1: Isolation of copper(I) piperidide (6) intermediate
    # Products: copper(I) piperi-dide (6) [tokens 8-10]
    # Reactants: copper(I) mesityl [14-15], piperidine [27]
    # Solvent: tet-rahydrofuran [21]
    # Temp: room temperature [35-36]
    # Time: 5 min [38-39]
    (
        "Isolation of the intermediate in the synthesis of copper(I) piperi-dide (6): "
        "A solution of copper(I) mesityl (365 mg, 2.00 mmol) in tet-rahydrofuran (2 mL) "
        "was treated with piperidine (988 mL, 10.00 mmol) and stirred at room temperature "
        "for 5 min after which a yellow precipitate was present.",
        {
             8: "B-PRODUCT",  9: "I-PRODUCT", 10: "I-PRODUCT",
            14: "B-REACTANT", 15: "I-REACTANT",
            21: "B-SOLVENT",
            27: "B-REACTANT",
            35: "B-TEMP",    36: "I-TEMP",
            38: "B-TIME",    39: "I-TIME",
        },
    ),

    # Para 2: Preparation of copper(I) piperidide (4)
    # Products: copper(I) piperidide (4) [tokens 2-4]
    # Reactants: copper(I) mesityl [8-9], piperidine [21]
    # Solvent: tetrahydrofuran [15]
    # Temp: room temperature [27-28]
    (
        "Preparation of copper(I) piperidide (4): A solution of copper(I) mesityl "
        "(1.462 g, 8.00 mmol) in tetrahydrofuran (10 mL) was treated with piperidine "
        "(3.95 mL, 40.00 mmol) at room temperature.",
        {
             2: "B-PRODUCT",  3: "I-PRODUCT",  4: "I-PRODUCT",
             8: "B-REACTANT", 9: "I-REACTANT",
            15: "B-SOLVENT",
            21: "B-REACTANT",
            27: "B-TEMP",    28: "I-TEMP",
        },
    ),

    # Para 3: Reaction with iodobenzene → arylamine (87% yield)
    # Solvent: [D6]DMSO [1]  (token includes trailing comma → still overlaps)
    # Reactants: copper(I) amide 4 [5-7], iodobenzene [9]
    # Products: arylamine [12-13], piperidine side-product [20-21]
    # Yield: 87 % [15-16], 8 % [24-25]
    (
        "In [D6]DMSO, the reaction of copper(I) amide 4 with iodobenzene gave the "
        "arylamine product in 87 % yield with some piperidine side-product also formed "
        "(8 % yield).",
        {
             1: "B-SOLVENT",
             5: "B-REACTANT", 6: "I-REACTANT", 7: "I-REACTANT",
             9: "B-REACTANT",
            12: "B-PRODUCT",  13: "I-PRODUCT",
            15: "B-YIELD",    16: "I-YIELD",
            20: "B-PRODUCT",  21: "I-PRODUCT",
            24: "B-YIELD",    25: "I-YIELD",
        },
    ),

    # Para 4: Adding reagent to copper(I) chloride suspension
    # Time: 10 min [3-4]  ("min," token still overlaps with span "10 min")
    # Reactant: copper(I) chloride [14-15]
    # Solvent: tetrahydrofuran [21]
    # Temp: room temperature [25-26]
    (
        "After stirring for 10 min, the solution was transferred dropwise to a suspension "
        "of copper(I) chloride (1.39 g, 14.00 mmol) in tetrahydrofuran (50 mL) at room temperature.",
        {
             3: "B-TIME",     4: "I-TIME",
            14: "B-REACTANT", 15: "I-REACTANT",
            21: "B-SOLVENT",
            25: "B-TEMP",    26: "I-TEMP",
        },
    ),

    # Para 5: Preparation of copper(I) benzylamide (5)
    # Products: copper(I) benzylamide (5) [tokens 2-4]
    # Reactants: copper(I) mesityl [8-9], benzylamine [21]
    # Solvent: tetrahydrofuran [15]
    # Temp: room temperature [27-28]
    (
        "Preparation of copper(I) benzylamide (5): A solution of copper(I) mesityl "
        "(439 mg, 2.40 mmol) in tetrahydrofuran (1 mL) was treated with benzylamine "
        "(288 mL, 2.64 mmol) at room temperature.",
        {
             2: "B-PRODUCT",  3: "I-PRODUCT",  4: "I-PRODUCT",
             8: "B-REACTANT", 9: "I-REACTANT",
            15: "B-SOLVENT",
            21: "B-REACTANT",
            27: "B-TEMP",    28: "I-TEMP",
        },
    ),
]


def build_sample(text: str, label_map: dict[int, str]) -> dict:
    tokens = _tokenize(text)
    labels = ["O"] * len(tokens)
    for idx, lbl in label_map.items():
        assert idx < len(tokens), f"Token index {idx} out of range ({len(tokens)} tokens)"
        labels[idx] = lbl
    return {"tokens": tokens, "labels": labels, "source": "gold_human"}


def main():
    out_dir = Path("data/training/gold_test")
    out_dir.mkdir(parents=True, exist_ok=True)

    samples = [build_sample(text, lmap) for text, lmap in GOLD_SAMPLES]

    # Verify no B-/I- sequence errors
    for i, s in enumerate(samples):
        prev = "O"
        for j, lbl in enumerate(s["labels"]):
            if lbl.startswith("I-"):
                entity = lbl[2:]
                if not (prev == f"B-{entity}" or prev == f"I-{entity}"):
                    print(f"[WARN] Sample {i+1} token {j}: I-{entity} after {prev!r} — possible label error")
            prev = lbl

    out = out_dir / "gold_chem0021.jsonl"
    with out.open("w", encoding="utf-8") as f:
        for s in samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")

    # Label list
    all_labels = {l for s in samples for l in s["labels"]}
    entity_labels = sorted({l[2:] for l in all_labels if l != "O"})
    label_list = ["O"] + [f"B-{l}" for l in entity_labels] + [f"I-{l}" for l in entity_labels]
    (out_dir / "labels.json").write_text(json.dumps(label_list, indent=2))

    print(f"[INFO] {len(samples)} gold samples → {out}")
    print(f"[INFO] Labels: {label_list}")
    for s in samples:
        non_o = [(t, l) for t, l in zip(s["tokens"], s["labels"]) if l != "O"]
        print(f"\n  '{s['tokens'][0]}...': {len(s['tokens'])} tokens, {len(non_o)} labeled")
        for tok, lbl in non_o:
            print(f"    {lbl:12s}  {tok!r}")


if __name__ == "__main__":
    main()
