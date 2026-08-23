"""
Persistent, user-editable prompt template store.

The JSON file at STORE_PATH is seeded from prompts.AVAILABLE_PROMPTS on first
run; from then on the file is the single source of truth, so templates
(including the built-in ones) can be added, edited, or removed through the UI
without touching prompts.py.
"""
import json
import os
from typing import List, Optional

from prompts import PromptTemplate, AVAILABLE_PROMPTS

STORE_PATH = "prompt_templates.json"


def _seed_data() -> List[dict]:
    return [
        {"name": t.name, "description": t.description, "system_prompt": t.system_prompt}
        for t in AVAILABLE_PROMPTS
    ]


def _write(data: List[dict]) -> None:
    with open(STORE_PATH, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def load_templates() -> List[PromptTemplate]:
    """Load templates from disk, seeding the store with the built-in defaults on first run"""
    if not os.path.exists(STORE_PATH):
        _write(_seed_data())

    with open(STORE_PATH, 'r', encoding='utf-8') as f:
        data = json.load(f)

    return [PromptTemplate(name=t["name"], description=t["description"], system_prompt=t["system_prompt"]) for t in data]


def save_templates(templates: List[PromptTemplate]) -> None:
    _write([
        {"name": t.name, "description": t.description, "system_prompt": t.system_prompt}
        for t in templates
    ])


def upsert_template(templates: List[PromptTemplate], template: PromptTemplate, original_name: Optional[str] = None) -> List[PromptTemplate]:
    """Add a new template, or update an existing one in place (matched by original_name when renaming)"""
    key = original_name or template.name
    updated = [t for t in templates if t.name != key]
    updated.append(template)
    save_templates(updated)
    return updated


def delete_template(templates: List[PromptTemplate], name: str) -> List[PromptTemplate]:
    updated = [t for t in templates if t.name != name]
    save_templates(updated)
    return updated
