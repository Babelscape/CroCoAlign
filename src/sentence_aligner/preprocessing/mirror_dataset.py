import argparse
import jsonlines
import json
import os
from tqdm import tqdm


class MirrorDataset:
    def __init__(self) -> None:
        pass

    def preprocess(self, filedir):
        for filename in tqdm(os.listdir(filedir)):
            samples = []
            file = open(f"{filedir}/{filename}", "r")
            lines = file.readlines()
            file.close()
            samples = []
            mirror_filename = (
                "test_" + filename[8:10] + filename[7] + filename[5:7] + filename[10:]
            )
            for line in lines:
                example = json.loads(line)
                sources_ids = example["sources"]["ids"]
                sources_text = example["sources"]["text"]
                sources_embs = example["sources"]["embs"]
                target_ids = example["targets"]["ids"]
                target_text = example["targets"]["text"]
                target_embs = example["targets"]["embs"]
                example["sources"]["ids"] = target_ids
                example["sources"]["text"] = target_text
                example["sources"]["embs"] = target_embs
                example["targets"]["ids"] = sources_ids
                example["targets"]["text"] = sources_text
                example["targets"]["embs"] = sources_embs
                samples.append(example)
            fw = jsonlines.open(f"{filedir}/{mirror_filename}", mode="w")
            for line in samples:
                fw.write(line)
            fw.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="DatasetPreprocess arguments usage: ", add_help=True
    )
    parser.add_argument(
        "filedir",
        help="Path to the file containing the ground truth (supported format: jsonl).",
    )
    args = parser.parse_args()
    Dp = MirrorDataset()
    Dp.preprocess(args.filedir)
