import jsonlines
import pandas as pd
import argparse
import ntpath
from datasets import Dataset, load_dataset
from collections import defaultdict


class DatasetGenerator:
    def __init__(self) -> None:
        pass

    def path_leaf(self, path):
        _, tail = ntpath.split(path)
        return tail

    def find_source_sentences(self, sources_ids, source_dict):
        text = []
        for sid in sources_ids:
            if sid != "":
                text.append(source_dict[sid])
        return {"ids": sources_ids, "text": text}

    def find_target_sentences(self, targets_ids, target_dict):
        text = []
        for tid in targets_ids:
            if tid != "":
                text.append(target_dict[tid])
        return {"ids": targets_ids, "text": text}

    def generate_dataset_from_xml(
        self, source_path, target_path, labels_path, slang, tlang
    ):
        df_source = pd.read_xml(source_path, xpath=".//s")
        df_target = pd.read_xml(target_path, xpath=".//s")
        df_labels = pd.read_xml(labels_path, xpath=".//link")

        ds_source = Dataset.from_pandas(df_source)
        ds_target = Dataset.from_pandas(df_target)
        ds_labels = Dataset.from_pandas(df_labels)

        source_dict = defaultdict()
        target_dict = defaultdict()

        for line in ds_source:
            source_dict[line["id"]] = line["s"]

        for line in ds_target:
            target_dict[line["id"]] = line["s"]

        result = jsonlines.open(
            slang + "-" + tlang + "-" + self.path_leaf(source_path)[0:-4] + ".jsonl",
            mode="w",
        )

        for line in ds_labels["xtargets"]:
            line = line.split(";")
            sources_ids = line[0].split(" ")
            targets_ids = line[1].split(" ")
            row = defaultdict()
            row["sources"] = self.find_source_sentences(sources_ids, source_dict)
            row["targets"] = self.find_target_sentences(targets_ids, target_dict)
            result.write(row)

        result.close()

    def generate_dataset_from_txt(
        self, source_path, target_path, labels_path, slang, tlang
    ):
        source_file = open(source_path, "r").readlines()
        target_file = open(target_path, "r").readlines()
        labels_file = open(labels_path, "r").readlines()

        source_dict = dict()
        target_dict = dict()

        source_id_list = []
        target_id_list = []

        for line in labels_file:
            line = line.strip().replace(" ", "")
            line = line.split(":")
            sources_ids = line[0][1:-1].split(",")
            targets_ids = line[1][1:-1].split(",")
            if sources_ids != [""] and sources_ids != []:
                for elem in sources_ids:
                    source_id_list.append(elem)
            if targets_ids != [""] and sources_ids != []:
                for elem in targets_ids:
                    target_id_list.append(elem)

        for sid, s in zip(source_id_list, source_file):
            source_dict[sid] = s.strip()

        for tid, t in zip(target_id_list, target_file):
            target_dict[tid] = t.strip()

        result = jsonlines.open(
            slang + "-" + tlang + "-" + self.path_leaf(source_path)[0:-3] + ".jsonl",
            mode="w",
        )

        for line in labels_file:
            line = line.strip().replace(" ", "")
            line = line.split(":")
            sources_ids = line[0][1:-1].split(",")
            targets_ids = line[1][1:-1].split(",")
            if sources_ids == []:
                sources_ids = [""]
            if targets_ids == []:
                targets_ids = [""]
            print(sources_ids, targets_ids)
            row = defaultdict()
            row["sources"] = self.find_source_sentences(sources_ids, source_dict)
            row["targets"] = self.find_target_sentences(targets_ids, target_dict)
            result.write(row)

        result.close()

    def generate_dataset_from_bleualign(
        self, source_path, target_path, labels_path, slang, tlang
    ):
        source_file = open(source_path, "r").readlines()
        target_file = open(target_path, "r").readlines()
        labels_file = open(labels_path, "r").readlines()

        source_dict = dict()
        target_dict = dict()
        labels = list()

        for line in labels_file:
            line = line.strip().replace(" ", "")
            line = line.split(":")
            sources_ids = line[0][1:-1].split(",")
            targets_ids = line[1][1:-1].split(",")
            sources_ids = tuple(sorted(sources_ids))
            targets_ids = tuple(sorted(targets_ids))
            labels.append([sources_ids, targets_ids])

        for i, s in enumerate(source_file):
            source_dict[str(i)] = s.strip()

        for j, t in enumerate(target_file):
            target_dict[str(j)] = t.strip()

        result = jsonlines.open(
            slang + "-" + tlang + "-" + self.path_leaf(source_path)[0:-3] + ".jsonl",
            mode="w",
        )

        for pair in labels:
            sources = pair[0]
            targets = pair[1]
            row = defaultdict()
            row["sources"] = {
                "ids": list(sources),
                "text": [source_dict[sid] for sid in sources if sid != ""],
            }
            row["targets"] = {
                "ids": list(targets),
                "text": [target_dict[tid] for tid in targets if tid != ""],
            }
            result.write(row)

        result.close()

    def generate_dataset(
        self, source_path, target_path, labels_path, slang, tlang, format
    ):
        if format == "xml":
            self.generate_dataset_from_xml(
                source_path, target_path, labels_path, slang, tlang
            )
        elif format == "bleualign":
            self.generate_dataset_from_bleualign(
                source_path, target_path, labels_path, slang, tlang
            )
        else:
            self.generate_dataset_from_txt(
                source_path, target_path, labels_path, slang, tlang
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="DatasetGenerator usage: ", add_help=True
    )
    parser.add_argument(
        "source_path",
        help="Path to the file containing the source file (supported format: .xml).",
    )
    parser.add_argument(
        "target_path",
        help="Path to the file containing the target file (supported format: .xml).",
    )
    parser.add_argument(
        "labels_path",
        help="Path to the file containing the ground truth (supported format: opus .tmp).",
    )
    parser.add_argument("--slang", nargs="?", help="Source language.")
    parser.add_argument("--tlang", nargs="?", help="Target language.")
    parser.add_argument(
        "--format", nargs="?", help="Format of the source and target files (.xml, .txt)"
    )
    args = parser.parse_args()
    DatasetGenerator().generate_dataset(
        args.source_path,
        args.target_path,
        args.labels_path,
        args.slang,
        args.tlang,
        args.format,
    )
