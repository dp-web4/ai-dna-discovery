
import json
import os

def load_index():
    path = os.path.join(os.path.dirname(__file__), "semantic_index.json")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def translate_glyph(glyph):
    index = load_index()
    for symbol in index["symbols"]:
        if symbol["glyph"] == glyph:
            return {
                "name": symbol["name"],
                "concept": symbol["concept"],
                "linked_meanings": symbol["linked_meanings"]
            }
    return {"error": "Glyph not found"}
