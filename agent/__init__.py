"""Chemical text annotation agent — LangGraph edition.

Public API
----------
run_pipeline(pdf_path, query=None) -> list[dict]
    Parse a PDF and return chemical entity annotations (8-label NER schema).

run_property_pipeline(pdf_path) -> list[dict]  [TODO]
    Parse a PDF and extract property records (5-role schema):
    substance / property_name / property_value / conditions / measurement_method
"""

from .graph import run_pipeline                    # noqa: F401
from .graph_property import run_property_pipeline  # noqa: F401

__all__ = ["run_pipeline", "run_property_pipeline"]
