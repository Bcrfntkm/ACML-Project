"""Prompt templates for property extraction (5-role schema).

Schema
------
Each measurement in a text is represented as a tuple of five roles:

  Substance          — the chemical compound whose property is measured
  Property_Name      — name of the property (e.g. solubility, melting point)
  Property_Value     — measured value with units (e.g. 15 mg/mL, 78 °C)
  Conditions         — experimental conditions (solvent, temperature, pH, …)
  Measurement_Method — technique used (shake-flask, DSC, Karl Fischer, …)

Any role that is not mentioned in the text should be omitted from the record.

Exports
-------
SYSTEM_PROMPT_PROPERTY   : str
FEW_SHOT_PROPERTY        : list[dict]  (role: user / assistant pairs)
build_property_messages(paragraph, accumulated_substances, rag_context, version)
    → list[BaseMessage]
"""

from __future__ import annotations

import json as _json

try:
    from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
except ImportError as exc:
    raise ImportError("pip install langchain-core") from exc


# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

SYSTEM_PROMPT_PROPERTY: str = (
    "You are an expert chemistry information-extraction assistant. "
    "Your task is to read a paragraph from a scientific paper and extract "
    "all property measurements it describes.\n\n"
    "For each measurement produce one JSON record with these fields "
    "(omit a field if the text does not mention it):\n"
    "  substance           — chemical compound whose property is measured\n"
    "  property_name       — name of the property (e.g. solubility, melting point, logP, yield, Rf)\n"
    "  property_value      — measured value with its unit (e.g. 15 mg/mL, 78 °C, 64%, 0.48)\n"
    "  conditions          — reaction or measurement conditions: catalyst, temperature, solvent "
    "used IN THE REACTION or measurement (e.g. Cu(OAc)2, DMSO, 125 °C, air)\n"
    "  measurement_method  — technique or instrument (e.g. shake-flask, DSC, HPLC, 1H NMR)\n\n"
    "Output format — ONLY valid JSON, no markdown fences, no commentary:\n"
    '  {"records": [{"substance": "...", "property_name": "...", '
    '"property_value": "...", "conditions": "...", "measurement_method": "..."}, ...]}\n\n'
    "Rules:\n"
    "• Extract every measurement mentioned, even if partial (missing conditions, etc.).\n"
    "• Copy values verbatim from the text — do not paraphrase or convert units.\n"
    "• If the same property is measured under different conditions, create one record per condition.\n"
    "• Do NOT invent values that are not explicitly stated in the paragraph.\n"
    "• In characterisation/NMR sections: extract yield, Rf (as property_name='Rf'), melting point, "
    "mass, physical state. For Rf, 'conditions' = TLC eluent (e.g. 'hexane:EtOAc 80:20').\n"
    "• 'conditions' must be reaction/measurement conditions — NEVER put a substance name in 'conditions'.\n"
    "• Deduplicate: if the exact same (substance, property_name, property_value) appears multiple times, "
    "output it only once.\n"
    "• Do NOT extract NMR spectral data: δ chemical shifts, coupling constants (J = … Hz), "
    "multiplicities (d, t, m, s, br), NMR frequency (300 MHz, 400 MHz), "
    "MS m/z values, HRMS data, or IR peak lists. These are structure-verification data, not property measurements.\n"
    "• Return an empty list if the paragraph contains no property measurements: "
    '{"records": []}'
)


# ---------------------------------------------------------------------------
# Few-shot examples
# ---------------------------------------------------------------------------

def _prop_answer(records: list[dict]) -> str:
    return _json.dumps({"records": records}, ensure_ascii=False)


# --- Example 1: solubility ---
_EX1_TEXT = (
    "The aqueous solubility of acetaminophen was determined to be 15 mg/mL "
    "at 25 °C using the shake-flask method. "
    "In ethanol, the solubility increased to 210 mg/mL under identical conditions."
)
_EX1_RECORDS = [
    {
        "substance":          "acetaminophen",
        "property_name":      "aqueous solubility",
        "property_value":     "15 mg/mL",
        "conditions":         "water, 25 °C",
        "measurement_method": "shake-flask method",
    },
    {
        "substance":          "acetaminophen",
        "property_name":      "solubility",
        "property_value":     "210 mg/mL",
        "conditions":         "ethanol, 25 °C",
        "measurement_method": "shake-flask method",
    },
]

# --- Example 2: melting point + yield ---
_EX2_TEXT = (
    "Compound 3a was isolated as a white solid (yield 82%, 56 mg) "
    "with a melting point of 143–145 °C (lit. 144 °C). "
    "Its pKa in water was measured by potentiometric titration and found to be 9.38."
)
_EX2_RECORDS = [
    {
        "substance":      "Compound 3a",
        "property_name":  "melting point",
        "property_value": "143–145 °C",
    },
    {
        "substance":          "Compound 3a",
        "property_name":      "pKa",
        "property_value":     "9.38",
        "conditions":         "water",
        "measurement_method": "potentiometric titration",
    },
]

# --- Example 3: reaction yield (property, not characterization) ---
_EX3_TEXT = (
    "The reaction of 2-phenylpyridine (1a, 0.5 mmol) with Cu(OAc)2 (1.0 equiv) "
    "in DMSO (2 mL) at 120 °C for 12 h under air afforded product 2a in 60% yield. "
    "Lowering the temperature to 80 °C reduced the yield to 34%."
)
_EX3_RECORDS = [
    {
        "substance":      "2a",
        "property_name":  "yield",
        "property_value": "60%",
        "conditions":     "DMSO, 120 °C, 12 h, air, Cu(OAc)2 (1.0 equiv)",
    },
    {
        "substance":      "2a",
        "property_name":  "yield",
        "property_value": "34%",
        "conditions":     "DMSO, 80 °C, 12 h, air, Cu(OAc)2 (1.0 equiv)",
    },
]

# --- Example 4: characterisation section (Rf, yield, physical state) ---
_EX4_TEXT = (
    "2-(2-(Methylthio)phenyl)pyridine (2a). "
    "Colorless liquid, yield 64% (42 mg), "
    "Rf (0.48) in hexane:EtOAc (80:20); "
    "1H NMR (300 MHz, CDCl3): δ 8.74 (d, J = 2.7 Hz, 1H), 7.80–7.75 (m, 1H)."
)
_EX4_RECORDS = [
    {
        "substance":      "2-(2-(Methylthio)phenyl)pyridine",
        "property_name":  "yield",
        "property_value": "64%",
    },
    {
        "substance":      "2-(2-(Methylthio)phenyl)pyridine",
        "property_name":  "mass",
        "property_value": "42 mg",
    },
    {
        "substance":      "2-(2-(Methylthio)phenyl)pyridine",
        "property_name":  "Rf",
        "property_value": "0.48",
        "conditions":     "hexane:EtOAc (80:20)",
        "measurement_method": "TLC",
    },
    # NOT extracted: NMR δ shifts, J constants, NMR frequency, compound code (2a)
]

# --- Example 5: no measurements → empty records ---
_EX5_TEXT = (
    "We thank Prof. Smith for helpful discussions. "
    "This work was supported by Grant No. 12345."
)
_EX5_RECORDS: list[dict] = []


def _make_prop_pair(text: str, records: list[dict]) -> list[dict]:
    return [
        {
            "role": "user",
            "content": (
                "=== EXTRACTION INSTRUCTION ===\n"
                "Extract all property measurements from the paragraph below.\n\n"
                "=== PARAGRAPH ===\n"
                f"{text}\n\n"
                "Return JSON ONLY: "
                '{"records": [{"substance": "...", "property_name": "...", '
                '"property_value": "...", "conditions": "...", '
                '"measurement_method": "..."}, ...]}'
            ),
        },
        {
            "role": "assistant",
            "content": _prop_answer(records),
        },
    ]


FEW_SHOT_PROPERTY: list[dict] = [
    *_make_prop_pair(_EX1_TEXT, _EX1_RECORDS),
    *_make_prop_pair(_EX2_TEXT, _EX2_RECORDS),
    *_make_prop_pair(_EX3_TEXT, _EX3_RECORDS),
    *_make_prop_pair(_EX4_TEXT, _EX4_RECORDS),
    *_make_prop_pair(_EX5_TEXT, _EX5_RECORDS),
]


# ---------------------------------------------------------------------------
# Message builder
# ---------------------------------------------------------------------------

def _few_shot_messages(examples: list[dict]) -> list[BaseMessage]:
    messages = []
    for ex in examples:
        if ex["role"] == "user":
            messages.append(HumanMessage(content=ex["content"]))
        elif ex["role"] == "assistant":
            messages.append(AIMessage(content=ex["content"]))
    return messages


def build_property_messages(
    paragraph: str,
    accumulated_substances: list[str] | None = None,
    rag_context: list[str] | None = None,
) -> list[BaseMessage]:
    """Build LangChain messages for property extraction from a single paragraph.

    Parameters
    ----------
    paragraph:
        The text to extract properties from.
    accumulated_substances:
        Substance names found in previous paragraphs — helps maintain consistency
        in naming across a multi-paragraph document.
    rag_context:
        Optional retrieved reference chunks (e.g. from a vector index).
    """
    messages: list[BaseMessage] = [SystemMessage(content=SYSTEM_PROMPT_PROPERTY)]
    messages.extend(_few_shot_messages(FEW_SHOT_PROPERTY))

    user_parts: list[str] = []

    if rag_context:
        ctx_block = "\n\n".join(
            f"[Reference chunk {i+1}]\n{chunk}" for i, chunk in enumerate(rag_context)
        )
        user_parts.append(
            "=== REFERENCE CONTEXT ===\n"
            "Use this context to help resolve ambiguous substance names or values, "
            "but only extract information that appears in the PARAGRAPH below.\n\n"
            f"{ctx_block}\n=== END REFERENCE CONTEXT ==="
        )

    if accumulated_substances:
        sub_list = ", ".join(f'"{s}"' for s in accumulated_substances[:30])
        user_parts.append(
            "=== SUBSTANCES SEEN IN PREVIOUS PARAGRAPHS ===\n"
            "Use consistent naming for these substances if they appear again.\n\n"
            f"{sub_list}\n=== END PREVIOUS SUBSTANCES ==="
        )

    user_parts.append(
        "=== EXTRACTION INSTRUCTION ===\n"
        "Extract all property measurements from the paragraph below."
    )
    user_parts.append(
        f"=== PARAGRAPH ===\n{paragraph}\n=== END PARAGRAPH ==="
    )
    user_parts.append(
        "Return JSON ONLY:\n"
        '{"records": [{"substance": "...", "property_name": "...", '
        '"property_value": "...", "conditions": "...", "measurement_method": "..."}, ...]}\n'
        "Omit any field that is not mentioned in the paragraph. "
        "Return {\"records\": []} if no measurements are present."
    )

    messages.append(HumanMessage(content="\n\n".join(user_parts)))
    return messages
