"""Prompt templates for the chemical annotation agent.

Exports
-------
SYSTEM_PROMPT, FEW_SHOT_EXAMPLES : current defaults (v2).
PROMPT_VERSIONS : dict mapping version string → {system_prompt, few_shot_examples}.
build_paragraph_messages(paragraph, query, accumulated_entities, rag_context, version)
build_messages(text, query, rag_context, version)  — whole-text variant (legacy).
"""

from __future__ import annotations

import json as _json

try:
    from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
except ImportError as exc:
    raise ImportError("pip install langchain-core") from exc


# ---------------------------------------------------------------------------
# Helper: build the JSON answer string for a few-shot example
# ---------------------------------------------------------------------------

def _fsa(text: str, spans: list[tuple[str, str]]) -> str:
    """few-shot answer: compute char offsets and return annotations JSON string."""
    annotations = []
    cursor = 0
    for span_text, label in spans:
        idx = text.find(span_text, cursor)
        if idx == -1:
            idx = text.find(span_text)  # retry from start (for repeated spans)
        if idx == -1:
            continue
        annotations.append({"text": span_text, "label": label,
                             "start": idx, "end": idx + len(span_text)})
        cursor = idx + len(span_text)
    return _json.dumps({"annotations": annotations}, ensure_ascii=False)


# ---------------------------------------------------------------------------
# System prompts
# ---------------------------------------------------------------------------

SYSTEM_PROMPT_V1: str = (
    "You are an expert chemistry annotator. Your task is to identify and "
    "annotate chemical entities in user-provided text according to the user's "
    "instructions.\n\n"
    "Recognised entity labels:\n"
    "  COMPOUND  — chemical compound names, molecular formulas, SMILES strings\n"
    "  IUPAC     — IUPAC systematic names\n"
    "  CAS       — CAS Registry numbers (e.g. 50-78-2)\n"
    "  REACTION  — chemical reactions or reaction names (e.g. acetylation)\n"
    "  PROPERTY  — physicochemical properties (e.g. melting point, yield, pKa)\n"
    "  ELEMENT   — chemical elements (e.g. carbon, Fe)\n"
    "  UNIT      — units of measurement attached to chemical quantities\n"
    "  OTHER     — any other chemistry-relevant span the user requests\n\n"
    "You MUST return ONLY a JSON object with a single key 'annotations' whose "
    "value is a list of objects of the form:\n"
    '  {"text": <exact substring>, "label": <UPPER_CASE_LABEL>, '
    '"start": <int char offset>, "end": <int char offset>}\n\n'
    "Rules:\n"
    "• The 'text' field MUST exactly match text[start:end] of the original input.\n"
    "• Offsets are 0-based, end-exclusive.\n"
    "• Do NOT include any commentary, explanations, or markdown — only valid JSON."
)

SYSTEM_PROMPT_V2: str = (
    "You are an expert chemistry annotator. Identify and annotate chemical "
    "entities in user-provided text according to the user's instructions.\n\n"
    "Recognised entity labels:\n"
    "  COMPOUND  — chemical compound names, molecular formulas, abbreviations, SMILES\n"
    "  IUPAC     — IUPAC systematic names\n"
    "  CAS       — CAS Registry numbers (format: \\d{2,7}-\\d{2}-\\d, e.g. 50-78-2)\n"
    "  REACTION  — reaction types or transformations (e.g. acetylation, C–H activation)\n"
    "  PROPERTY  — physicochemical properties and measurements "
    "(yield %, melting point, Rf, pKa, mass, volume, time, temperature, pressure)\n"
    "  ELEMENT   — chemical elements (e.g. carbon, Fe, Pd)\n"
    "  UNIT      — standalone units of measurement (e.g. °C, mmol, mL)\n"
    "  OTHER     — other chemistry-relevant spans explicitly requested\n\n"
    "Output format — ONLY valid JSON, no markdown fences, no commentary:\n"
    '  {"annotations": [{"text": <exact substring>, "label": <LABEL>, '
    '"start": <int>, "end": <int>}, ...]}\n\n'
    "Offset rules:\n"
    "• Offsets are 0-based, end-exclusive; text[start:end] must exactly equal 'text'.\n"
    "• Nested spans are allowed: annotate both the outer compound name AND an inner "
    "IUPAC name if both appear. Example: 'aspirin (acetylsalicylic acid)' → "
    "COMPOUND span for 'aspirin' + IUPAC span for 'acetylsalicylic acid'.\n\n"
    "Characterisation / NMR sections:\n"
    "• DO annotate: compound names, yield (e.g. 'yield 64%'), mass ('42 mg'), "
    "Rf values ('Rf (0.48)'), NMR solvents ('CDCl3', 'DMSO-d6') as COMPOUND.\n"
    "• DO NOT annotate: NMR δ chemical shift values, coupling constants (J = … Hz), "
    'multiplicities (d, t, m, s), NMR frequency ("300 MHz"), compound codes like (2a).\n\n'
    "Do NOT annotate:\n"
    "• Citation/reference numbers: [1], [1,2], superscript numbers.\n"
    "• Footnote markers, section numbers, page numbers.\n"
    "• Standalone stoichiometry coefficients without chemical context."
)

SYSTEM_PROMPT_V3: str = (
    "You are an expert chemistry annotator. Identify and annotate chemical "
    "entities in user-provided text according to the user's instructions.\n\n"
    "ALLOWED LABELS — you MUST use ONLY these exact strings in the 'label' field:\n"
    "  COMPOUND  — chemical compound names, molecular formulas, abbreviations, SMILES\n"
    "  IUPAC     — IUPAC systematic names (full IUPAC nomenclature)\n"
    "  CAS       — CAS Registry numbers (format: digits-digits-digit, e.g. 50-78-2)\n"
    "  REACTION  — reaction types or transformations (e.g. acetylation, C–H activation)\n"
    "  PROPERTY  — physicochemical properties and measurements "
    "(yield %, melting point, Rf, pKa, mass, volume, time, temperature, pressure)\n"
    "  ELEMENT   — chemical elements (e.g. carbon, Fe, Pd, copper)\n"
    "  UNIT      — standalone units of measurement (e.g. °C, mmol, mL, Hz)\n"
    "  OTHER     — any other chemistry-relevant span not covered above\n\n"
    "CRITICAL: The 'label' value MUST be exactly one of the 8 strings above. "
    "NEVER use compound names, element symbols, or any other string as a label. "
    "For example: Cu(II) complexes → label=COMPOUND, not label='CU(II)'. "
    "NMR nuclei (1H, 13C) → label=OTHER, not label='1H'.\n\n"
    "Output format — ONLY valid JSON, no markdown fences, no commentary:\n"
    '  {"annotations": [{"text": <exact substring>, "label": <LABEL>, '
    '"start": <int>, "end": <int>}, ...]}\n\n'
    "Offset rules:\n"
    "• Offsets are 0-based, end-exclusive; text[start:end] must exactly equal 'text'.\n"
    "• For context continuity: if an entity was mentioned earlier in the document, "
    "use the same label when it appears again.\n\n"
    "Characterisation / NMR sections:\n"
    "• DO annotate: compound names, yield ('yield 64%'), mass ('42 mg'), "
    "Rf values ('Rf (0.48)'), NMR solvents ('CDCl3', 'DMSO-d6') as COMPOUND.\n"
    "• DO NOT annotate: NMR δ shift values, J coupling constants, "
    "multiplicities (d, t, m, s), NMR frequency ('300 MHz'), compound codes like (2a).\n\n"
    "Do NOT annotate:\n"
    "• Citation/reference numbers: [1], [1,2], superscript numbers.\n"
    "• Footnote markers, section numbers, page numbers.\n"
    "• Standalone stoichiometry coefficients without chemical context."
)


# ---------------------------------------------------------------------------
# Few-shot examples
# ---------------------------------------------------------------------------

# --- Example A: simple compound + CAS + property ---
_EX_A_TEXT = (
    "Copper(II) acetate (CAS 142-71-2) was dissolved in ethanol at 60 °C. "
    "The reaction yielded a blue precipitate with 85% yield."
)
_EX_A_SPANS = [
    ("Copper(II) acetate", "COMPOUND"),
    ("CAS 142-71-2",       "CAS"),
    ("ethanol",            "COMPOUND"),
    ("60 °C",              "PROPERTY"),
    ("85% yield",          "PROPERTY"),
]

# --- Example B: IUPAC + reaction ---
_EX_B_TEXT = (
    "The synthesis of aspirin (acetylsalicylic acid, C9H8O4) involves the "
    "acetylation of salicylic acid with acetic anhydride."
)
_EX_B_SPANS = [
    ("aspirin",            "COMPOUND"),
    ("acetylsalicylic acid","IUPAC"),
    ("C9H8O4",             "COMPOUND"),
    ("acetylation",        "REACTION"),
    ("salicylic acid",     "COMPOUND"),
    ("acetic anhydride",   "COMPOUND"),
]

# --- Example C (v2 only): NMR characterisation paragraph ---
_EX_C_TEXT = (
    "2-(2-(Methylthio)phenyl)pyridine (2a). "
    "Colorless liquid, yield 64% (42 mg), "
    "Rf (0.48) in hexane:EtOAc (80:20); "
    "1H NMR (300 MHz, CDCl3): "
    "δ 8.74 (d, J = 2.7 Hz, 1H), 7.80–7.75 (m, 1H)."
)
_EX_C_SPANS = [
    ("2-(2-(Methylthio)phenyl)pyridine", "COMPOUND"),
    ("Colorless liquid",                 "PROPERTY"),
    ("yield 64%",                        "PROPERTY"),
    ("42 mg",                            "PROPERTY"),
    ("Rf (0.48)",                        "PROPERTY"),
    ("hexane",                           "COMPOUND"),
    ("EtOAc",                            "COMPOUND"),
    ("CDCl3",                            "COMPOUND"),
    # NOT annotated: "300 MHz", δ values, J constants, (2a), multiplicities
]

# --- Example D (v2 only): reaction conditions / scope ---
_EX_D_TEXT = (
    "2-Phenylpyridine (1a, 0.5 mmol) was treated with Cu(OAc)2 (1.0 equiv) "
    "in DMSO (2 mL) at 120 °C for 12 h under air to give compound 2a "
    "in 60% yield. "
    "Using DMF as solvent reduced the yield to 25%. "
    "CuCl2 and Cu(OTf)2 were ineffective."
)
_EX_D_SPANS = [
    ("2-Phenylpyridine", "COMPOUND"),
    ("0.5 mmol",         "PROPERTY"),
    ("Cu(OAc)2",         "COMPOUND"),
    ("1.0 equiv",        "PROPERTY"),
    ("DMSO",             "COMPOUND"),
    ("2 mL",             "PROPERTY"),
    ("120 °C",      "PROPERTY"),
    ("12 h",             "PROPERTY"),
    ("60% yield",        "PROPERTY"),
    ("DMF",              "COMPOUND"),
    ("25%",              "PROPERTY"),
    ("CuCl2",            "COMPOUND"),
    ("Cu(OTf)2",         "COMPOUND"),
    # NOT annotated: "(1a)" compound code, "2a" compound code, "air"
]


def _make_few_shot_pair(text: str, spans: list[tuple[str, str]]) -> list[dict]:
    """Return [user_turn_dict, assistant_turn_dict] for a few-shot example."""
    return [
        {
            "role": "user",
            "content": (
                "=== ANNOTATION INSTRUCTION ===\n"
                "Annotate all chemical entities.\n\n"
                "=== TEXT TO ANNOTATE ===\n"
                f"{text}\n\n"
                "Return JSON ONLY in the format: "
                '{"annotations": [{"text": "...", "label": "...", "start": 0, "end": 0}, ...]}.'
            ),
        },
        {
            "role": "assistant",
            "content": _fsa(text, spans),
        },
    ]


FEW_SHOT_EXAMPLES_V1: list[dict] = [
    *_make_few_shot_pair(_EX_A_TEXT, _EX_A_SPANS),
    *_make_few_shot_pair(_EX_B_TEXT, _EX_B_SPANS),
]

FEW_SHOT_EXAMPLES_V2: list[dict] = [
    *_make_few_shot_pair(_EX_A_TEXT, _EX_A_SPANS),
    *_make_few_shot_pair(_EX_B_TEXT, _EX_B_SPANS),
    *_make_few_shot_pair(_EX_C_TEXT, _EX_C_SPANS),  # NMR characterisation
    *_make_few_shot_pair(_EX_D_TEXT, _EX_D_SPANS),  # reaction conditions/scope
]


# ---------------------------------------------------------------------------
# Version registry
# ---------------------------------------------------------------------------

PROMPT_VERSIONS: dict[str, dict] = {
    "v1": {
        "system_prompt":     SYSTEM_PROMPT_V1,
        "few_shot_examples": FEW_SHOT_EXAMPLES_V1,
        "description":       "Original: 2 few-shot examples, basic rules.",
    },
    "v2": {
        "system_prompt":     SYSTEM_PROMPT_V2,
        "few_shot_examples": FEW_SHOT_EXAMPLES_V2,
        "description": (
            "Improved: nesting rule, NMR guidance, negative examples, "
            "4 few-shot examples (compound/CAS, IUPAC/reaction, NMR section, "
            "reaction scope/conditions)."
        ),
    },
    "v3": {
        "system_prompt":     SYSTEM_PROMPT_V3,
        "few_shot_examples": FEW_SHOT_EXAMPLES_V2,
        "description": (
            "v3: explicit label whitelist + error examples to prevent label hallucination. "
            "Same 4 few-shot examples as v2."
        ),
    },
}

# Defaults used by existing code that imports these names directly
SYSTEM_PROMPT:     str        = SYSTEM_PROMPT_V2
FEW_SHOT_EXAMPLES: list[dict] = FEW_SHOT_EXAMPLES_V2


# ---------------------------------------------------------------------------
# Message builders
# ---------------------------------------------------------------------------

def _few_shot_messages(examples: list[dict]) -> list[BaseMessage]:
    messages = []
    for ex in examples:
        if ex["role"] == "user":
            messages.append(HumanMessage(content=ex["content"]))
        elif ex["role"] == "assistant":
            messages.append(AIMessage(content=ex["content"]))
    return messages


def build_paragraph_messages(
    paragraph: str,
    query: str,
    accumulated_entities: list[str],
    rag_context: list[str],
    version: str = "v2",
) -> list[BaseMessage]:
    """Build the LangChain message list for paragraph-level annotation.

    Parameters
    ----------
    version:
        Prompt version key from ``PROMPT_VERSIONS`` (default: "v2").
        Pass "v1" to use the original prompts — useful for ablation experiments.
    """
    cfg = PROMPT_VERSIONS.get(version, PROMPT_VERSIONS["v2"])
    messages: list[BaseMessage] = [SystemMessage(content=cfg["system_prompt"])]
    messages.extend(_few_shot_messages(cfg["few_shot_examples"]))

    user_parts: list[str] = []

    if rag_context:
        ctx_block = "\n\n".join(
            f"[Reference chunk {i+1}]\n{chunk}" for i, chunk in enumerate(rag_context)
        )
        user_parts.append(
            "=== REFERENCE CONTEXT ===\n"
            "Use the following reference material to inform your annotations, "
            "but only label spans that appear in the PARAGRAPH TO ANNOTATE.\n\n"
            f"{ctx_block}\n=== END REFERENCE CONTEXT ==="
        )

    if accumulated_entities:
        entity_list = ", ".join(f'"{e}"' for e in accumulated_entities[:30])
        user_parts.append(
            "=== ENTITIES FOUND IN PREVIOUS PARAGRAPHS ===\n"
            "Use this list to maintain label consistency — if you see the same "
            "entity again, use the same label.\n\n"
            f"{entity_list}\n=== END PREVIOUS ENTITIES ==="
        )

    user_parts.append(f"=== ANNOTATION INSTRUCTION ===\n{query}")
    user_parts.append(
        f"=== PARAGRAPH TO ANNOTATE ===\n{paragraph}\n=== END PARAGRAPH ==="
    )
    user_parts.append(
        "Return JSON ONLY in the format:\n"
        '{"annotations": [{"text": "...", "label": "...", "start": 0, "end": 0}, ...]}\n'
        "Offsets are relative to the start of this paragraph (0-based). "
        "Only annotate spans that appear verbatim in the paragraph above."
    )

    messages.append(HumanMessage(content="\n\n".join(user_parts)))
    return messages


def build_messages(
    text: str,
    query: str,
    rag_context: list[str],
    version: str = "v2",
) -> list[BaseMessage]:
    """Whole-text variant (legacy). Prefer build_paragraph_messages for long texts."""
    cfg = PROMPT_VERSIONS.get(version, PROMPT_VERSIONS["v2"])
    messages: list[BaseMessage] = [SystemMessage(content=cfg["system_prompt"])]
    messages.extend(_few_shot_messages(cfg["few_shot_examples"]))

    user_parts: list[str] = []

    if rag_context:
        ctx_block = "\n\n".join(
            f"[Context chunk {i+1}]\n{chunk}" for i, chunk in enumerate(rag_context)
        )
        user_parts.append(
            "You are provided with retrieved reference context. "
            "Use it to inform your annotations, but only label spans from the TEXT TO ANNOTATE.\n\n"
            f"=== RETRIEVED CONTEXT ===\n{ctx_block}\n=== END CONTEXT ==="
        )

    user_parts.append(f"=== ANNOTATION INSTRUCTION ===\n{query}")
    user_parts.append(f"=== TEXT TO ANNOTATE ===\n{text}")
    user_parts.append(
        "Return JSON ONLY in the format: "
        '{"annotations": [{"text": "...", "label": "...", "start": 0, "end": 0}, ...]}.'
    )

    messages.append(HumanMessage(content="\n\n".join(user_parts)))
    return messages
