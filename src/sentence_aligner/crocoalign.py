import argparse
import os
import math
from pathlib import Path
import jsonlines
from itertools import combinations

import numpy as np
import torch
import torch.nn as nn
import faiss
from nn_core.serialization import load_model
from pytorch_lightning.utilities import move_data_to_device
from transformers import PreTrainedTokenizer, AutoTokenizer
from sentence_aligner.pl_modules.pl_module import MyLightningModule


class TestEvaluator:
    def __init__(self, model_path, dataset_dir, batch_size=16):
        self.device = "cuda"
        self.model = (
            load_model(
                module_class=MyLightningModule,
                checkpoint_path=Path(model_path),
                map_location=self.device,
            )
            .to(self.device)
            .eval()
        )
        self.dataset_dir = dataset_dir
        self.transformer_name = "sentence-transformers/LaBSE"
        self.tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(
            self.transformer_name
        )
        self.sentence_encoder = self.model.transformer
        self.batch_size = batch_size
        self.k = 200
        self.sid2stext = dict()
        self.tid2ttext = dict()
        self.sindex2sid = dict()
        self.tindex2tid = dict()
        self.sid2sindex = dict()
        self.tid2tindex = dict()
        self.source_result = dict()
        self.target_result = dict()
        self.sources_emb_matrix = []
        self.targets_emb_matrix = []

    def join_target_text_combinations(self, combination):
        return " ".join(combination)

    def encode_samples(self, sources_samples, targets_samples):
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

    def evaluate(self, sources, targets):
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

            ## Aumentare K e filtrare per percentuale distanza libro
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

        self.labse_batch_misalignments_recovery(sources, targets)

    def labse_batch_misalignments_recovery(self, sources, targets):
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

    def cluster_result(self):
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

    def write_results_file(self, book_dir, book_name, source_lang, target_lang):
        fout = jsonlines.open(
            self.dataset_dir
            + "/"
            + book_dir
            + "/"
            + "aligned_"
            + book_name
            + f".{source_lang}"
            + f".{target_lang}"
            + ".jsonl",
            "w",
        )
        for k, v in self.source_result.items():
            row = dict()
            row["sources"] = {"ids": [k], "text": [self.sid2stext[k]]}
            row["targets"] = {"ids": v, "text": [self.tid2ttext[t] for t in v]}
            fout.write(row)
        fout.close()

    def main(self):
        books_dir = os.listdir(self.dataset_dir)
        for book_dir in books_dir:
            source_file, target_file, source_lang, target_lang = "", "", "", ""
            for file_as_source in os.listdir(self.dataset_dir + "/" + book_dir):
                if file_as_source.endswith(".jsonl"):
                    continue
                source_file = os.path.join(self.dataset_dir, book_dir, file_as_source)
                source_lang = file_as_source.split(".")[-1]
                for file_as_target in os.listdir(self.dataset_dir + "/" + book_dir):
                    if file_as_target.endswith(".jsonl"):
                        continue
                    target_file = os.path.join(
                        self.dataset_dir, book_dir, file_as_target
                    )
                    target_lang = file_as_target.split(".")[-1]
                    if file_as_source != file_as_target and source_lang != target_lang:
                        book_name = source_file.split("/")[-1].split(".")[0]
                        print(
                            f"CURRENT TEST FILE: {source_lang} -> {target_lang}: {book_name}"
                        )

                        self.sid2stext = dict()
                        self.tid2ttext = dict()
                        self.sindex2sid = dict()
                        self.tindex2tid = dict()
                        self.sid2sindex = dict()
                        self.tid2tindex = dict()
                        self.source_result = dict()
                        self.target_result = dict()
                        self.sources_emb_matrix = []
                        self.targets_emb_matrix = []
                        sources = []
                        targets = []
                        sources_emb_list = []
                        targets_emb_list = []

                        source_dataset = open(source_file, "r")
                        target_dataset = open(target_file, "r")

                        for i, line in enumerate(source_dataset):
                            s = line.strip()
                            sid = "s" + str(i)
                            source_emb = self.sentence_encoder.encode(
                                s, show_progress_bar=False
                            )
                            sources.append(
                                dict(
                                    sources_ids=[sid],
                                    sources=[s],
                                    sources_embs=[source_emb],
                                )
                            )
                            self.source_result.setdefault(str(sid), [])
                            self.sindex2sid[str(i)] = sid
                            self.sid2sindex[str(sid)] = i
                            self.sid2stext[str(sid)] = s
                            sources_emb_list.append(source_emb)

                        for j, line in enumerate(target_dataset):
                            t = line.strip()
                            tid = "t" + str(j)
                            target_emb = self.sentence_encoder.encode(
                                t, show_progress_bar=False
                            )
                            targets.append(
                                dict(
                                    targets_ids=[tid],
                                    targets=[t],
                                    targets_embs=[target_emb],
                                )
                            )
                            self.target_result.setdefault(str(tid), [])
                            self.tindex2tid[str(j)] = tid
                            self.tid2tindex[str(tid)] = j
                            self.tid2ttext[str(tid)] = t
                            targets_emb_list.append(target_emb)

                        self.sources_emb_matrix = np.array(
                            sources_emb_list, dtype=np.float32
                        )
                        self.targets_emb_matrix = np.array(
                            targets_emb_list, dtype=np.float32
                        )

                        with torch.no_grad():
                            self.evaluate(sources, targets)

                        self.cluster_result()

                        self.write_results_file(
                            book_dir, book_name, source_lang, target_lang
                        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="TestEvaluator arguments usage: ", add_help=True
    )
    parser.add_argument("model_path", help="Path to the model checkpoint.")
    parser.add_argument(
        "test_dataset_dir", help="Path to the test dataset to be aligned."
    )
    parser.add_argument(
        "--batch_size", nargs="?", type=int, default=16, help="Batch size."
    )
    args = parser.parse_args()
    evaluator = TestEvaluator(args.model_path, args.test_dataset_dir, args.batch_size)
    evaluator.main()
