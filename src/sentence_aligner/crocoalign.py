import argparse
import csv
import os
import time
from collections import defaultdict
import math
from pathlib import Path
import jsonlines
from itertools import combinations
from typing import List, Dict, Tuple
import numpy as np
import torch
import torch.nn as nn
import json
import faiss
from nn_core.serialization import load_model
from pytorch_lightning.utilities import move_data_to_device
from transformers import PreTrainedTokenizer, AutoTokenizer
from sentence_transformers import SentenceTransformer
from sentence_aligner.pl_modules.pl_module import MyLightningModule


class CroCoAlign:
    def __init__(
        self,
        ckpt_path: str,
        source_file: str,
        target_file: str,
        precomputed_embeddings: bool,
        batch_size=32,
        recovery="labse-batch",
        out="tsv",
    ):
        self.device: str = "cuda" if torch.cuda.is_available() else "cpu"
        self.precomputed_embeddings: bool = precomputed_embeddings
        self.recovery: str = recovery
        self.fileformat: str = out
        self.model: MyLightningModule = (
            load_model(
                module_class=MyLightningModule,
                checkpoint_path=Path(ckpt_path),
                map_location=self.device,
            )
            .to(self.device)
            .eval()
        )
        self.source_file: str = source_file
        self.target_file: str = target_file
        self.sentence_transformer_name: str = "sentence-transformers/LaBSE"
        self.tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(
            self.sentence_transformer_name
        )
        self.sentence_encoder: SentenceTransformer = self.model.transformer
        self.batch_size: int = batch_size
        self.k: int = 200
        self.sid2stext: dict = dict()
        self.tid2ttext: dict = dict()
        self.sindex2sid: dict = dict()
        self.tindex2tid: dict = dict()
        self.sid2sindex: dict = dict()
        self.tid2tindex: dict = dict()
        self.source_result: dict = dict()
        self.target_result: dict = dict()
        self.sources_emb_matrix: list = []
        self.targets_emb_matrix: list = []

    def encode_samples(
        self,
        sources_samples: List[Dict[str, List]],
        targets_samples: List[Dict[str, List]],
    ) -> Dict:
        """
        Collate function to create a batch from the current source and target samples

        Args:
            sources_samples: List[Dict[str, List]] of source samples
            targets_samples: List[Dict[str, List]] of target samples

        Returns:
            output: Dict containing the batched input data
        """

        sources_ids = [g for s in sources_samples for g in s["sources_ids"]]
        targets_ids = [g for s in targets_samples for g in s["targets_ids"]]

        sources = [g for s in sources_samples for g in s["sources"]]
        targets = [g for s in targets_samples for g in s["targets"]]

        sources_encodings = self.tokenizer(
            sources,
            truncation=True,
            padding=True,
            return_tensors="pt",
            return_token_type_ids=True,
        )

        target_encodings = self.tokenizer(
            targets,
            return_tensors="pt",
            padding=True,
            truncation=True,
            return_offsets_mapping=True,
            return_token_type_ids=True,
        )

        sources_embeds = torch.stack(
            [torch.tensor(g) for s in sources_samples for g in s["sources_embs"]]
        )
        targets_embeds = torch.stack(
            [torch.tensor(g) for s in targets_samples for g in s["targets_embs"]]
        )

        output = dict()
        output["sources_text"] = sources
        output["targets_text"] = targets
        output["sources_ids"] = sources_ids
        output["targets_ids"] = targets_ids
        output["sources_encodings"] = sources_encodings
        output["targets_encodings"] = target_encodings
        output["sources_embeds"] = sources_embeds
        output["targets_embeds"] = targets_embeds

        return output

    def evaluate(
        self, sources: List[Dict[str, List]], targets: List[Dict[str, List]]
    ) -> None:
        """
        Evaluate the model on the given sources and targets sentences contained in the input documents.

        Args:
            sources: List[Dict[str, List]] of source samples
            targets: List[Dict[str, List]] of target samples

        Note:
            - min_dist = 0.05 is used to discard the target sentences that have an high relative distance
              with respect to the source sentences.
            - self.k regulates the number of nearest neighbors to be retrieved from the target sentences
        """
        for i in range(math.ceil(len(sources) / self.batch_size)):
            source_start = i * self.batch_size
            source_context_start = max(0, source_start - (self.batch_size))
            source_context_end = min(source_start + (self.batch_size), len(sources))
            source_batch = sources[source_context_start:source_context_end]

            index = faiss.index_factory(
                self.sources_emb_matrix.shape[1], "Flat", faiss.METRIC_INNER_PRODUCT
            )
            faiss.normalize_L2(self.sources_emb_matrix[source_start : source_start + 1])
            faiss.normalize_L2(self.targets_emb_matrix)
            index.add(self.targets_emb_matrix)

            _, I = index.search(
                self.sources_emb_matrix[source_start : source_start + 1], self.k
            )

            targets_starts = [int(index) for index in I[0] if int(index) != -1]

            min_dist = 0.05
            targets_starts_filtered = []
            for ts in targets_starts:
                curr_dist = np.abs(source_start / len(sources) - ts / len(targets))
                if curr_dist < min_dist:
                    targets_starts_filtered.append(ts)

            targets_starts = targets_starts_filtered

            max_num_alignments = 0
            best_target_batch = []

            for target_start in targets_starts:
                target_context_start = max(0, target_start - (self.batch_size))
                target_context_end = min(target_start + (self.batch_size), len(targets))
                target_batch = targets[target_context_start:target_context_end]

                batch = self.encode_samples(source_batch, target_batch)

                batch = move_data_to_device(batch, device=self.device)
                step_out = self.model.step(batch, 0, split="test", compute_loss=False)
                output = dict(
                    predictions=step_out["predictions"],
                    matrix_index=step_out["matrix_index"],
                    sources_text=batch["sources_text"],
                    targets_text=batch["targets_text"],
                    sources_ids=batch["sources_ids"],
                    targets_ids=batch["targets_ids"],
                )

                preds = torch.round(output["predictions"].flatten())

                prediciton_indices = output["matrix_index"][preds.bool()]
                sources_ids = output["sources_ids"]
                targets_ids = output["targets_ids"]

                prediction_couples = dict()

                num_preds = int(torch.sum(preds))

                if num_preds > max_num_alignments:
                    max_num_alignments = num_preds
                    best_target_batch = target_batch

            batch = self.encode_samples(source_batch, best_target_batch)

            batch = move_data_to_device(batch, device=self.device)
            step_out = self.model.step(batch, 0, split="test", compute_loss=False)
            output = dict(
                predictions=step_out["predictions"],
                matrix_index=step_out["matrix_index"],
                sources_text=batch["sources_text"],
                targets_text=batch["targets_text"],
                sources_ids=batch["sources_ids"],
                targets_ids=batch["targets_ids"],
            )

            preds = torch.round(output["predictions"].flatten())

            prediciton_indices = output["matrix_index"][preds.bool()]
            sources_ids = output["sources_ids"]
            targets_ids = output["targets_ids"]

            prediction_couples = dict()

            for s_idx, t_idx in prediciton_indices.detach().tolist():
                prediction_couples.setdefault(s_idx, []).append(t_idx)
            for s_idx, t_idx in prediction_couples.items():
                t_ids = [targets_ids[t] for t in t_idx]
                for t in t_ids:
                    if t not in self.source_result[sources_ids[s_idx]]:
                        self.source_result[sources_ids[s_idx]].append(t)
                    if sources_ids[s_idx] not in self.target_result[t]:
                        self.target_result[t].append(sources_ids[s_idx])

        if self.recovery == "labse":
            self.labse_misalignments_recovery()
        elif self.recovery == "labse-batch":
            self.labse_batch_misalignments_recovery(sources, targets)

    def labse_misalignments_recovery(self) -> None:
        """
        Recovery procedure to handle misalignments in the source result
        by means of the LaBSE sentence embeddings.
        """
        cosine_similarity = nn.CosineSimilarity(dim=0, eps=1e-6)
        for k, v in self.source_result.items():
            if len(v) > 1:
                v_indexes = sorted([self.tid2tindex[vi] for vi in v])
                if not (v_indexes == list(range(min(v_indexes), max(v_indexes) + 1))):
                    sim = -1
                    source_emb = self.sentence_encoder.encode(
                        self.sid2stext[k], show_progress_bar=False
                    )

                    v_indexes_combs = []
                    for r in range(1, len(v_indexes) + 1):
                        v_indexes_combs.extend(
                            list(map(list, combinations(v_indexes, r)))
                        )
                    v_indexes_combs = [
                        elem
                        for elem in v_indexes_combs
                        if sorted(elem) == list(range(min(elem), max(elem) + 1))
                    ]

                    targets_ids_combs = []
                    for elem in v_indexes_combs:
                        res = []
                        for index in elem:
                            res.append(self.tindex2tid[str(index)])
                        targets_ids_combs.append(res)

                    targets_text_combs = []
                    for elem in targets_ids_combs:
                        res = []
                        for tid in elem:
                            res.append(self.tid2ttext[tid])
                        targets_text_combs.append(res)

                    targets_text_combs = [" ".join(elem) for elem in targets_text_combs]

                    targets_embs = self.sentence_encoder.encode(
                        targets_text_combs, show_progress_bar=False
                    )
                    for target_ids_comb, target_emb in zip(
                        targets_ids_combs, targets_embs
                    ):
                        cs = cosine_similarity(
                            torch.tensor(source_emb), torch.tensor(target_emb)
                        )
                        if cs > sim:
                            sim = cs
                            self.source_result[k] = target_ids_comb

    def labse_batch_misalignments_recovery(
        self, sources: List[Dict[str, List]], targets: List[Dict[str, List]]
    ) -> None:
        """
        Recovery procedure to handle misalignments in the source result
        by means of the LaBSE sentence embeddings + the batched strategy

        Args:
            sources: List[Dict[str, List]] of source samples
            targets: List[Dict[str, List]] of target samples
        """
        cosine_similarity = nn.CosineSimilarity(dim=0, eps=1e-6)
        for k, v in self.source_result.items():
            if len(v) > 1:
                v_indexes = sorted([self.tid2tindex[vi] for vi in v])
                if not (v_indexes == list(range(min(v_indexes), max(v_indexes) + 1))):
                    sim = -1
                    source_start = self.sid2sindex[k]
                    source_context_start = max(0, source_start - 3)
                    source_context_end = min(source_start + 3, len(sources))
                    source_batch = sources[source_context_start:source_context_end]
                    source_sentences = [elem["sources"][0] for elem in source_batch]
                    source_text_comb = " ".join(source_sentences)

                    source_emb = self.sentence_encoder.encode(
                        source_text_comb, show_progress_bar=False
                    )

                    v_indexes_combs = []
                    for r in range(1, len(v_indexes) + 1):
                        v_indexes_combs.extend(
                            list(map(list, combinations(v_indexes, r)))
                        )
                    v_indexes_combs = [
                        elem
                        for elem in v_indexes_combs
                        if sorted(elem) == list(range(min(elem), max(elem) + 1))
                    ]

                    true_targets_ids_combs = []
                    for elem in v_indexes_combs:
                        res = []
                        for index in elem:
                            res.append(self.tindex2tid[str(index)])
                        true_targets_ids_combs.append(res)

                    new_v_indexes_combs = []
                    for elem in v_indexes_combs:
                        new_elem = elem
                        new_elem.extend(
                            [
                                idx
                                for idx in list(
                                    range(
                                        max(0, min(new_elem) - 3),
                                        min(max(new_elem) + 3, len(targets)),
                                    )
                                )
                            ]
                        )
                        new_v_indexes_combs.append(sorted(list(set(new_elem))))

                    v_indexes_combs = new_v_indexes_combs

                    targets_ids_combs = []
                    for elem in v_indexes_combs:
                        res = []
                        for index in elem:
                            res.append(self.tindex2tid[str(index)])
                        targets_ids_combs.append(res)

                    targets_text_combs = []
                    for elem in targets_ids_combs:
                        res = []
                        for tid in elem:
                            res.append(self.tid2ttext[tid])
                        targets_text_combs.append(res)

                    targets_text_combs = [" ".join(elem) for elem in targets_text_combs]

                    targets_embs = self.sentence_encoder.encode(
                        targets_text_combs, show_progress_bar=False
                    )
                    for target_ids_comb, target_emb in zip(
                        true_targets_ids_combs, targets_embs
                    ):
                        cs = cosine_similarity(
                            torch.tensor(source_emb), torch.tensor(target_emb)
                        )
                        if cs > sim:
                            sim = cs
                            self.source_result[k] = target_ids_comb

    def cluster_result(self) -> None:
        """
        Function to cluster the results: for each source sentence which shares a target sentence
        with another source sentence, join their predictions.
        """
        values_dict = dict()
        cluster_result = dict()
        final_result = dict()

        for _, values in self.source_result.items():
            for vi in values:
                for vj in values:
                    values_dict.setdefault(vi, []).append(vj)

        for keyi, valuesi in values_dict.items():
            new_value = set(valuesi)
            for _, valuesj in values_dict.items():
                if len(new_value & set(valuesj)) > 0:
                    new_value = new_value | set(valuesj)
            cluster_result.setdefault(keyi, []).extend(sorted(list(new_value)))

        for key, values in self.source_result.items():
            final_result.setdefault(key, [])
            for vi in values:
                cluster = cluster_result[vi]
                final_result[key].extend(cluster)
            final_result[key] = list(dict.fromkeys(final_result[key]))

        self.source_result = final_result

    def write_results_file(self, source_filename: str, target_filename: str, fileformat: str) -> None:
        """
        Function to write the results into a file.

        Args:
            source_filename: Name of the source file
            target_filename: Name of the target file
            fileformat: Format of the file to be written
        """
        source_filename = source_filename.split("/")[-1].split(".")[0]
        target_filename = target_filename.split("/")[-1].split(".")[0]
        result_filename = f"results_{source_filename}_{target_filename}"
        if fileformat == "jsonl":
            out = jsonlines.open(result_filename + ".jsonl", "w")
            for k, v in self.source_result.items():
                row = dict()
                row["sources"] = {"ids": [k], "text": [self.sid2stext[k]]}
                row["targets"] = {"ids": v, "text": [self.tid2ttext[t] for t in v]}
                out.write(row)
            out.close()
        else:
            out = open(result_filename + ".tsv", "w")
            writer = csv.writer(out, delimiter="\t")
            writer.writerow(
                ["sources_ids", "source_sentences", "targets_ids", "target_sentences"]
            )
            for k, v in self.source_result.items():
                stext = self.sid2stext[k]
                ttext = [self.tid2ttext[t] for t in v]
                row = [k, stext, v, ttext]
                writer.writerow(row)
            out.close()

    def main(self):
        print(f"Aligning source file {self.source_file} with target file {self.target_file}")
        start_time = time.time()
        self.sid2stext = dict()
        self.tid2ttext = dict()
        self.sindex2sid = dict()
        self.tindex2tid = dict()
        self.sid2sindex = dict()
        self.tid2tindex = dict()
        self.source_result = dict()
        self.target_result = dict()
        self.ground_truth = dict()
        self.sources_emb_matrix = []
        self.targets_emb_matrix = []
        sources = []
        targets = []
        sources_emb_list = []
        targets_emb_list = []

        source_file = open(self.source_file, "r")
        source_lines = source_file.readlines()
        source_file.close()
        target_file = open(self.target_file, "r")
        target_lines = target_file.readlines()
        target_file.close()

        i = 0
        j = 0

        if self.precomputed_embeddings:
            for source_line in source_lines:
                source_example = json.loads(source_line)
                for sid, s, e in zip(
                    source_example["ids"],
                    source_example["text"],
                    source_example["embs"],
                ):
                    sources.append(
                        dict(sources_ids=[sid], sources=[s], sources_embs=[e])
                    )
                    self.source_result.setdefault(str(sid), [])
                    self.sindex2sid[str(i)] = sid
                    self.sid2sindex[str(sid)] = i
                    sources_emb_list.append(e)
                    self.sid2stext[str(sid)] = s
                    i += 1
            for target_line in target_lines:
                target_example = json.loads(target_line)
                for tid, t, e in zip(
                    target_example["ids"],
                    target_example["text"],
                    target_example["embs"],
                ):
                    targets.append(
                        dict(targets_ids=[tid], targets=[t], targets_embs=[e])
                    )
                    self.target_result.setdefault(str(tid), [])
                    self.tindex2tid[str(j)] = tid
                    self.tid2tindex[str(tid)] = j
                    targets_emb_list.append(e)
                    self.tid2ttext[str(tid)] = t
                    j += 1

            self.sources_emb_matrix = np.array(sources_emb_list, dtype=np.float32)
            self.targets_emb_matrix = np.array(targets_emb_list, dtype=np.float32)

        else:
            source_sentences = []
            target_sentences = []
            for source_line in source_lines:
                source_example = json.loads(source_line)
                for sid, s in zip(
                    source_example["ids"], 
                    source_example["text"]
                ):
                    source_sentences.append(s)
                    self.source_result.setdefault(str(sid), [])
                    self.sindex2sid[str(i)] = sid
                    self.sid2sindex[str(sid)] = i
                    self.sid2stext[str(sid)] = s
                    i += 1
            for target_line in target_lines:
                target_example = json.loads(target_line)
                for tid, t in zip(
                    target_example["ids"], 
                    target_example["text"]
                ):
                    target_sentences.append(t)
                    self.target_result.setdefault(str(tid), [])
                    self.tindex2tid[str(j)] = tid
                    self.tid2tindex[str(tid)] = j
                    self.tid2ttext[str(tid)] = t
                    j += 1

            source_embeddings = self.sentence_encoder.encode(
                source_sentences, batch_size=512
            ).tolist()
            target_embeddings = self.sentence_encoder.encode(
                target_sentences, batch_size=512
            ).tolist()
            for source_line in source_lines:
                source_example = json.loads(source_line)
                for sid, s in zip(
                    source_example["ids"], 
                    source_example["text"]
                ):
                    sources.append(
                        dict(
                            sources_ids=[sid],
                            sources=[s],
                            sources_embs=[
                                source_embeddings[self.sid2sindex[str(sid)]]
                            ],
                        )
                    )
            for target_line in target_lines:
                target_example = json.loads(target_line)
                for tid, t in zip(
                    target_example["ids"],
                    target_example["text"]
                ):
                    targets.append(
                        dict(
                            targets_ids=[tid],
                            targets=[t],
                            targets_embs=[
                                target_embeddings[self.tid2tindex[str(tid)]]
                            ],
                        )
                    )

            self.sources_emb_matrix = np.array(source_embeddings, dtype=np.float32)
            self.targets_emb_matrix = np.array(target_embeddings, dtype=np.float32)

        with torch.no_grad():
            self.evaluate(sources, targets)

        self.cluster_result()

        end_time = time.time()
        tike_taken = end_time - start_time
        print(f"TIME TAKEN: {tike_taken} seconds")

        self.write_results_file(self.source_file, self.target_file, self.fileformat)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="CroCoAlign arguments usage: ", add_help=True
    )
    parser.add_argument("ckpt_path", help="Path to the model checkpoint.")
    parser.add_argument(
        "source_file",
        help="""
        Path to the source file to be aligned. 
        Expects a jsonl file where each row is a dictionary with the following structure:
            - ids
            - text
            - embs (Optional)
        
        Tip: see the example file in the data directory.
        """
    )
    parser.add_argument(
        "target_file",
        help="""
        Path to the target file to be aligned. 
        Expects a jsonl file where each row is a dictionary with the following structure:
            - ids
            - text
            - embs (Optional)
        
        Tip: see the example file in the data directory.
        """
    )
    parser.add_argument(
        "--precomputed_embeddings",
        "-p",
        action="store_true",
        default=False,
        help="Flag specifying if the data to be aligned already contains precomputed embeddings for source and target sentences. If set to False, the model will compute the embeddings for the sentences. Default is False.",
    )
    parser.add_argument(
        "--batch_size", "-b", nargs="?", type=int, default=32, help="Batch size."
    )
    parser.add_argument(
        "--recovery",
        "-r",
        nargs="?",
        default="labse-batch",
        help="Type of misalignments recovery procedure. (labse, labse-batch).",
    )
    parser.add_argument(
        "--out", "-o", nargs="?", help="Results output file format. (jsonl, tsv)."
    )
    args = parser.parse_args()
    crocoalign = CroCoAlign(
        args.ckpt_path,
        args.source_file,
        args.target_file,
        args.precomputed_embeddings,
        args.batch_size,
        args.recovery,
        args.out,
    )
    crocoalign.main()