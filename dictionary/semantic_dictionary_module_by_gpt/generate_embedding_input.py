
from translator_core import load_index

def get_embedding_input():
    index = load_index()
    return [
        f"{entry['glyph']} = {entry['concept']}"
        for entry in index["symbols"]
    ]

if __name__ == "__main__":
    print("\n".join(get_embedding_input()))
