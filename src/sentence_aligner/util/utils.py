import json
from pathlib import Path
from typing import Dict, List, MutableMapping, Tuple

from tqdm import tqdm

char2pos: Dict[str, str] = dict(v="VERB", a="ADJ", n="NOUN", r="ADV")


class GlossManager(MutableMapping):
    def __init__(self, data_path: Path, *args, **kwargs):
        self.store = dict()
        self.update(dict(*args, **kwargs))  # use the free update to set keys
        self._load(data_path)

    def _load(self, data_path: Path):
        if data_path.exists():
            with data_path.open("r") as fr:
                for line in tqdm(fr, desc=f"Loading glosses from: {str(data_path)}"):
                    line = line.strip()
                    entry = json.loads(line)
                    synset_id = entry["synsetID"]
                    source = entry["source"]
                    gloss = entry["gloss"]
                    self.store.setdefault((synset_id, source), []).append(gloss)

    def __getitem__(self, key):
        return self.store[key]

    def __setitem__(self, key, value):
        self.store[key] = value

    def __delitem__(self, key):
        del self.store[key]

    def __iter__(self):
        return iter(self.store)

    def __len__(self):
        return len(self.store)


def load_inventory(path: Path) -> Dict[Tuple[str, str], List[str]]:
    word2candidates = dict()
    with path.open("r") as fr:
        for line in fr:
            line = line.strip()
            word_pos, *senses = line.split("\t")
            word, pos = word_pos.split("#")
            word2candidates[(word, pos)] = senses
    return word2candidates
