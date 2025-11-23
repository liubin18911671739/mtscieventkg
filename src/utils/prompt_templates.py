"""
Prompt templates and JSON schema for multilingual scientific event extraction.
"""

SCIEVENT_JSON_SCHEMA = {
    "type": "array",
    "items": {
        "type": "object",
        "properties": {
            "problem": {"type": "string", "description": "Scientific problem or task"},
            "method": {"type": "array", "items": {"type": "string"}},
            "data_or_material": {"type": "array", "items": {"type": "string"}},
            "finding": {"type": "string", "description": "Key empirical or theoretical finding"},
            "application": {"type": "array", "items": {"type": "string"}},
            "year": {"type": "integer"},
            "confidence": {"type": "number"}
        },
        "required": ["problem", "method", "data_or_material", "finding", "application"]
    }
}

SCIEVENT_PROMPT = """
You are an expert assistant for extracting scientific contribution events from multilingual papers (English/中文/Français/Español).
Return only JSON that strictly follows the schema. No explanations, no markdown.
If a field is missing, keep it as an empty string/array but maintain valid JSON.
Prefer concise phrases over long sentences; split multiple methods/applications into lists.

Schema:
{schema}

Paper metadata:
Title: {title}
Abstract: {abstract}
Language: {lang}
Publication year: {year}

Return only a JSON array, nothing else.
"""


def build_prompt(title: str, abstract: str, lang: str, year: int | str) -> str:
    """Fill the prompt template."""
    return SCIEVENT_PROMPT.format(
        schema=SCIEVENT_JSON_SCHEMA,
        title=title,
        abstract=abstract,
        lang=lang,
        year=year,
    )
