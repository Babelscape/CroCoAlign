import argparse
import jsonlines
import json
import os
import time
from tqdm import tqdm
from sentence_transformers import SentenceTransformer

class DatasetPreprocess:
    def __init__(self) -> None:
        self.device = "cuda"
        self.sentence_encoder = SentenceTransformer("sentence-transformers/LaBSE").to(
            self.device
        )

    def preprocess(self, filedir, outdir):
        avg_time = 0
        for filename in tqdm(os.listdir(filedir)):
            start_time = time.time()
            samples = []
            file = open(f"{filedir}/{filename}", "r")
            lines = file.readlines()
            file.close()
            source_sentences = []
            target_sentences = []
            sid2sindex = dict()
            tid2tindex = dict()
            i = 0
            j = 0
            for line in lines:
                example = json.loads(line)
                for sid, s in zip(
                    example["sources"]["ids"], example["sources"]["text"]
                ):
                    source_sentences.append(s)
                    sid2sindex[sid] = i
                    i += 1
                for tid, t in zip(
                    example["targets"]["ids"], example["targets"]["text"]
                ):
                    target_sentences.append(t)
                    tid2tindex[tid] = j
                    j += 1
            source_embeddings = self.sentence_encoder.encode(
                source_sentences, batch_size=512
            ).tolist()
            target_embeddings = self.sentence_encoder.encode(
                target_sentences, batch_size=512
            ).tolist()
            for line in lines:
                example = json.loads(line)
                example["sources"]["embs"] = []
                example["targets"]["embs"] = []
                for sid, s in zip(
                    example["sources"]["ids"], example["sources"]["text"]
                ):
                    example["sources"]["embs"].append(
                        source_embeddings[sid2sindex[sid]]
                    )
                for tid, t in zip(
                    example["targets"]["ids"], example["targets"]["text"]
                ):
                    example["targets"]["embs"].append(
                        target_embeddings[tid2tindex[tid]]
                    )
                samples.append(example)
            fw = jsonlines.open(f"{outdir}/{filename}", mode="w")
            for line in samples:
                fw.write(line)
            fw.close()
            end_time = time.time()
            time_taken = end_time - start_time
            avg_time += time_taken
            print(f"Time taken ", time_taken, " seconds")
        avg_time /= len(os.listdir(filedir))
        print(f"Average time taken ", avg_time, " seconds")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="DatasetPreprocess arguments usage: ", add_help=True
    )
    parser.add_argument(
        "filedir",
        "-i",
        help="Path to the file containing the data to be preprocessed (supported format: jsonl).",
    )
    parser.add_argument(
        "outdir",
        "-o",
        help="Path to the file containing the data to be preprocessed (supported format: jsonl).",
    )
    args = parser.parse_args()
    Dp = DatasetPreprocess()
    Dp.preprocess(args.filedir, args.outdir)
