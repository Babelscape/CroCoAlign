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


class TestEvaluator:
    def __init__(
        self,
        ckpt_path: str,
        data_dir: str,
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
        self.data_dir: Path = Path(data_dir)
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
        self.ground_truth: dict = dict()
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

    def _precision(self, goldalign: List[Tuple], testalign: List[Tuple]) -> np.ndarray:
        """
        Computes tpstrict, fpstrict, tplax, fplax for gold/test alignments
        Original evaluation code available at: https://github.com/thompsonb/vecalign/blob/ca96a30716f12241e14f836b06705107c771987c/score.py#L35

        Args:
            goldalign: List[Tuple] of gold alignments
            testalign: List[Tuple] of test alignments

        Retruns:
            np.ndarray: Array containing the tpstrict, fpstrict, tplax, fplax
        """
        tpstrict = 0  # true positive strict counter
        tplax = 0  # true positive lax counter
        fpstrict = 0  # false positive strict counter
        fplax = 0  # false positive lax counter

        # convert to sets, remove alignments empty on both sides
        testalign = set(
            [(tuple(x), tuple(y)) for x, y in testalign if len(x) or len(y)]
        )
        goldalign = set(
            [(tuple(x), tuple(y)) for x, y in goldalign if len(x) or len(y)]
        )

        # mappings from source test sentence idxs to
        #    target gold sentence idxs for which the source test sentence
        #    was found in corresponding source gold alignment
        src_id_to_gold_tgt_ids = defaultdict(set)
        for gold_src, gold_tgt in goldalign:
            for gold_src_id in gold_src:
                for gold_tgt_id in gold_tgt:
                    src_id_to_gold_tgt_ids[gold_src_id].add(gold_tgt_id)

        for test_src, test_target in testalign:
            if (test_src, test_target) == ((), ()):
                continue
            if (test_src, test_target) in goldalign:
                # strict match
                tpstrict += 1
                tplax += 1
            else:
                # For anything with partial gold/test overlap on the source,
                # see if there is also partial overlap on the gold/test target
                # If so, its a lax match
                target_ids = set()
                for src_test_id in test_src:
                    for tgt_id in src_id_to_gold_tgt_ids[src_test_id]:
                        target_ids.add(tgt_id)
                if set(test_target).intersection(target_ids):
                    fpstrict += 1
                    tplax += 1
                else:
                    fpstrict += 1
                    fplax += 1

        return np.array([tpstrict, fpstrict, tplax, fplax], dtype=np.int32)

    def score_multiple(
        self,
        gold_list: List[Dict[Tuple, Tuple]],
        test_list: List[Dict[Tuple, Tuple]],
        value_for_div_by_0=0.0,
    ) -> Dict:
        """
        Compute the precision, recall and f1 score (strict and lax) for the given gold and test alignments.
        Original code available at: https://github.com/thompsonb/vecalign/blob/ca96a30716f12241e14f836b06705107c771987c/score.py#L82

        Args:
            gold_list: List[Dict[Tuple, Tuple]] of gold alignments
            test_list: List[Dict[Tuple, Tuple]] of test alignments
            value_for_div_by_0: Value to be used in case of division by 0

        Returns:
            result: Dict containing the precision, recall and f1 score (strict and lax)
        """
        # accumulate counts for all gold/test files
        pcounts = np.array([0, 0, 0, 0], dtype=np.int32)
        rcounts = np.array([0, 0, 0, 0], dtype=np.int32)
        for goldalign, testalign in zip(gold_list, test_list):
            if isinstance(goldalign, dict):
                goldalign = list(goldalign.items())
            if isinstance(testalign, dict):
                testalign = list(testalign.items())
            pcounts += self._precision(goldalign=goldalign, testalign=testalign)
            # recall is precision with no insertion/deletion and swap args
            test_no_del = [(x, y) for x, y in testalign if len(x) and len(y)]
            gold_no_del = [(x, y) for x, y in goldalign if len(x) and len(y)]
            rcounts += self._precision(goldalign=test_no_del, testalign=gold_no_del)

        # Compute results
        # pcounts: tpstrict,fnstrict,tplax,fnlax
        # rcounts: tpstrict,fpstrict,tplax,fplax

        if pcounts[0] + pcounts[1] == 0:
            pstrict = value_for_div_by_0
        else:
            pstrict = pcounts[0] / float(pcounts[0] + pcounts[1])

        if pcounts[2] + pcounts[3] == 0:
            plax = value_for_div_by_0
        else:
            plax = pcounts[2] / float(pcounts[2] + pcounts[3])

        if rcounts[0] + rcounts[1] == 0:
            rstrict = value_for_div_by_0
        else:
            rstrict = rcounts[0] / float(rcounts[0] + rcounts[1])

        if rcounts[2] + rcounts[3] == 0:
            rlax = value_for_div_by_0
        else:
            rlax = rcounts[2] / float(rcounts[2] + rcounts[3])

        if (pstrict + rstrict) == 0:
            fstrict = value_for_div_by_0
        else:
            fstrict = 2 * (pstrict * rstrict) / (pstrict + rstrict)

        if (plax + rlax) == 0:
            flax = value_for_div_by_0
        else:
            flax = 2 * (plax * rlax) / (plax + rlax)

        result = dict(
            recall_strict=rstrict,
            recall_lax=rlax,
            precision_strict=pstrict,
            precision_lax=plax,
            f1_strict=fstrict,
            f1_lax=flax,
        )

        return result

    def compute_results(self) -> Dict:
        """
        Function to convert the self.source_result and self.ground_truth dictionaries into the format required by
        the self.score_multiple function.

        Returns:
            results: Dict containing the precision, recall and f1 score (strict and lax)
        """
        gt_tid2sid = dict()
        crocoalign_tid2sid = dict()
        keys = list(self.ground_truth.keys())

        for sid in keys:
            gt_target_ids = tuple(sorted(self.ground_truth[sid]))
            crocoalign_target_ids = tuple(sorted(self.source_result[sid]))
            if gt_target_ids not in gt_tid2sid:
                gt_tid2sid[gt_target_ids] = [sid]
            else:
                gt_tid2sid[gt_target_ids].append(sid)
            if crocoalign_target_ids not in crocoalign_tid2sid:
                crocoalign_tid2sid[crocoalign_target_ids] = [sid]
            else:
                crocoalign_tid2sid[crocoalign_target_ids].append(sid)

        gt_sid2tid = {tuple(sorted(v)): k for k, v in gt_tid2sid.items() if len(k) > 0}
        for k, v in gt_tid2sid.items():
            if len(k) == 0:
                for sid in v:
                    gt_sid2tid[tuple([sid])] = tuple([])
        crocoalign_sid2tid = {
            tuple(sorted(v)): k
            for k, v in crocoalign_tid2sid.items()
            if len(k) > 0 and k != tuple([""])
        }
        for k, v in crocoalign_tid2sid.items():
            if len(k) == 0 or k == tuple([""]):
                for sid in v:
                    crocoalign_sid2tid[tuple([sid])] = tuple([])

        results = self.score_multiple([gt_sid2tid], [crocoalign_sid2tid])
        return results

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

    def sanity_check(self) -> None:
        """
        Function to perform a sanity check on the results.
        The function checks if the model has correctly made as assignment for each source and target sentence.
        """
        print(f"CroCoAlign N° KEYS: {len(self.source_result.keys())}")
        print(f"GROUND TRUTH N° KEYS: {len(self.ground_truth.keys())}")
        a = set(self.source_result.keys())
        b = set(self.ground_truth.keys())
        diff = b.difference(a)
        diff2 = a.difference(b)
        print(f"DIFF GT - CroCoAlign (sid): {[k for k in list(diff)]}")
        print(f"DIFF CroCoAlign - GT (sid): {[k for k in list(diff2)]}")

    def write_results_file(self, filename: str, fileformat: str) -> None:
        """
        Function to write the results into a file.

        Args:
            filename: Name of the file to be written
            fileformat: Format of the file to be written
        """
        if fileformat == "jsonl":
            out = jsonlines.open("results_" + filename, "w")
            for k, v in self.source_result.items():
                row = dict()
                row["sources"] = {"ids": [k], "text": [self.sid2stext[k]]}
                row["targets"] = {"ids": v, "text": [self.tid2ttext[t] for t in v]}
                out.write(row)
            out.close()
        else:
            out = open("results_" + filename.split(".")[0] + ".tsv", "w")
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
        test_files = os.listdir(self.data_dir)
        avg_time = 0
        avg_result = defaultdict(float)
        for tf in test_files:
            start_time = time.time()
            test_data_path = os.path.join(self.data_dir, Path(tf))
            print(f"CURRENT TEST FILE: {tf}")
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

            dataset = open(test_data_path, "r")
            lines = dataset.readlines()

            i = 0
            j = 0

            if self.precomputed_embeddings:
                for line in lines:
                    example = json.loads(line)
                    for sid, s, e in zip(
                        example["sources"]["ids"],
                        example["sources"]["text"],
                        example["sources"]["embs"],
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
                    for tid, t, e in zip(
                        example["targets"]["ids"],
                        example["targets"]["text"],
                        example["targets"]["embs"],
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
                    for sid in example["sources"]["ids"]:
                        if sid != "":
                            self.ground_truth.setdefault(str(sid), [])
                            for tid in example["targets"]["ids"]:
                                if tid != "":
                                    self.ground_truth[str(sid)].append(tid)

                self.sources_emb_matrix = np.array(sources_emb_list, dtype=np.float32)
                self.targets_emb_matrix = np.array(targets_emb_list, dtype=np.float32)

            else:
                source_sentences = []
                target_sentences = []
                for line in lines:
                    example = json.loads(line)
                    for sid, s in zip(
                        example["sources"]["ids"], example["sources"]["text"]
                    ):
                        source_sentences.append(s)
                        self.source_result.setdefault(str(sid), [])
                        self.sindex2sid[str(i)] = sid
                        self.sid2sindex[str(sid)] = i
                        self.sid2stext[str(sid)] = s
                        i += 1
                    for tid, t in zip(
                        example["targets"]["ids"], example["targets"]["text"]
                    ):
                        target_sentences.append(t)
                        self.target_result.setdefault(str(tid), [])
                        self.tindex2tid[str(j)] = tid
                        self.tid2tindex[str(tid)] = j
                        self.tid2ttext[str(tid)] = t
                        j += 1
                    for sid in example["sources"]["ids"]:
                        if sid != "":
                            self.ground_truth.setdefault(str(sid), [])
                            for tid in example["targets"]["ids"]:
                                if tid != "":
                                    self.ground_truth[str(sid)].append(tid)
                source_embeddings = self.sentence_encoder.encode(
                    source_sentences, batch_size=512
                ).tolist()
                target_embeddings = self.sentence_encoder.encode(
                    target_sentences, batch_size=512
                ).tolist()
                for line in lines:
                    example = json.loads(line)
                    for sid, s in zip(
                        example["sources"]["ids"], example["sources"]["text"]
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
                    for tid, t in zip(
                        example["targets"]["ids"], example["targets"]["text"]
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

            self.sanity_check()

            self.cluster_result()

            end_time = time.time()
            tike_taken = end_time - start_time
            avg_time += tike_taken
            print(f"TIME TAKEN: {tike_taken} seconds")

            results = self.compute_results()
            avg_result["f1_strict"] += results["f1_strict"]
            avg_result["f1_lax"] += results["f1_lax"]
            print(f"{tf} RESULTS: {results}")

            self.write_results_file(tf, self.fileformat)

        avg_time = avg_time / len(test_files)
        avg_result["f1_strict"] = avg_result["f1_strict"] / len(test_files)
        avg_result["f1_lax"] = avg_result["f1_lax"] / len(test_files)
        print(f"AVG TIME TAKEN: {avg_time} seconds")
        print(
            f"AVG RESULTS: STRICT F1 {avg_result['f1_strict']}, LAX F1 {avg_result['f1_lax']}"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="CroCoAlign evaluation arguments usage: ", add_help=True
    )
    parser.add_argument("ckpt_path", help="Path to the model checkpoint.")
    parser.add_argument(
        "data_dir",
        help="Path to the directory containing the test data to be aligned. Expects a directory containing jsonl files.",
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
    evaluator = TestEvaluator(
        args.ckpt_path,
        args.data_dir,
        args.precomputed_embeddings,
        args.batch_size,
        args.recovery,
        args.out,
    )
    evaluator.main()
