import argparse
import ntpath


class DatasetPreprocess:
    def __init__(self) -> None:
        pass

    def path_leaf(self, path):
        _, tail = ntpath.split(path)
        return tail

    def preprocess(self, input_file, test_flag):
        filename = self.path_leaf(input_file)
        file = open(input_file)
        lines = file.readlines()
        train = []
        val = []
        test = []
        if test_flag:
            for line in lines:
                if len(val) < (0.5 * len(lines)):
                    val.append(line)
                else:
                    test.append(line)
            self.save_dataset("val", val, filename)
            self.save_dataset("test", test, filename)
        else:
            for line in lines:
                train.append(line)
            self.save_dataset("train", train, filename)
        file.close()

    def save_dataset(self, split_type, samples, filename):
        fw = open(
            f"/mnt/data/neural-sentence-aligner/data/opus/books/{split_type}/{split_type}_{filename}",
            "w",
        )
        for line in samples:
            fw.write(line)
        fw.close()

    def initialize(self):
        parser = argparse.ArgumentParser(
            description="DatasetPreprocess arguments usage: ", add_help=True
        )
        parser.add_argument(
            "filepath",
            help="Path to the file containing the ground truth (supported format: jsonl).",
        )
        parser.add_argument(
            "-t",
            "--test",
            action="store_true",
            help="Flag telling if the file is to be used as a test set (supported format: jsonl).",
        )
        args = parser.parse_args()

        input_file = args.filepath
        test_flag = args.test

        self.preprocess(input_file, test_flag)


if __name__ == "__main__":
    DatasetPreprocess().initialize()
