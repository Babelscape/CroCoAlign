import pandas as pd
import argparse
import ntpath
import json
import os


class FileConverter:
    def __init__(self) -> None:
        pass

    def path_leaf(self, path):
        _, tail = ntpath.split(path)
        return tail

    def from_jsonl_to_txt(self, filedir):
        for filename in os.listdir(filedir):
            fin = open(filedir + filename, "r")
            source_txt_file = open(filedir + filename[:-6] + "-source.txt", "w")
            target_txt_file = open(filedir + filename[:-6] + "-target.txt", "w")
            # source_txt_id_list = open(filedir + filename[:-6] + '-id-list.txt', "w")
            # target_txt_id_list = open(filedir + filename[:-6] + '-id-list.txt', "w")

            for line in fin:
                example = json.loads(line)
                for sid, stext in zip(
                    example["sources"]["ids"], example["sources"]["text"]
                ):
                    source_txt_file.write(stext)
                    source_txt_file.write("\n")
                    # source_txt_id_list.write(sid)
                    # source_txt_id_list.write("\n")
                for tid, ttext in zip(
                    example["targets"]["ids"], example["targets"]["text"]
                ):
                    target_txt_file.write(ttext)
                    target_txt_file.write("\n")
                    # target_txt_id_list.write(tid)
                    # target_txt_id_list.write("\n")

            source_txt_file.close()
            # source_txt_id_list.close()
            target_txt_file.close()
            # target_txt_id_list.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="FileConverter (from xml to txt) arguments usage: ", add_help=True
    )
    parser.add_argument(
        "filedir",
        help="Path to the directory containing the files to be converted to .txt. Supported formats: xml, jsonl.",
    )
    args = parser.parse_args()
    FileConverter().from_jsonl_to_txt(args.filedir)
