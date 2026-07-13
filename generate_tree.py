
import os

IGNORE = {".git", "__pycache__", "venv", "data/raw", "data/processed", "data/splits", "model"}

def print_tree(root=".", prefix=""):
    entries = sorted(os.listdir(root))
    entries = [e for e in entries if e not in {".git", "__pycache__", "venv"}]
    for i, entry in enumerate(entries):
        path = os.path.join(root, entry)
        connector = "└── " if i == len(entries) - 1 else "├── "
        print(prefix + connector + entry + ("/" if os.path.isdir(path) else ""))
        if os.path.isdir(path) and entry not in {"raw", "processed", "splits"}:
            extension = "    " if i == len(entries) - 1 else "│   "
            print_tree(path, prefix + extension)

print_tree(".")
