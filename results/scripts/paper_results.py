import os
import csv
import numpy as np
from collections import defaultdict
import ast
from argparse import ArgumentParser

class PaperResults:
    def __init__(self, results_dir):
        self.results_dir = results_dir

    def _precision(self, goldalign, testalign):
        """
        Computes tpstrict, fpstrict, tplax, fplax for gold/test alignments
        """
        tpstrict = 0  # true positive strict counter
        tplax = 0     # true positive lax counter
        fpstrict = 0  # false positive strict counter
        fplax = 0     # false positive lax counter

        # convert to sets, remove alignments empty on both sides
        testalign = set([(tuple(x), tuple(y)) for x, y in testalign if len(x) or len(y)])
        goldalign = set([(tuple(x), tuple(y)) for x, y in goldalign if len(x) or len(y)])

        # mappings from source test sentence idxs to
        #    target gold sentence idxs for which the source test sentence 
        #    was found in corresponding source gold alignment
        src_id_to_gold_tgt_ids = defaultdict(set)
        for gold_src, gold_tgt in goldalign:
            for gold_src_id in gold_src:
                for gold_tgt_id in gold_tgt:
                    src_id_to_gold_tgt_ids[gold_src_id].add(gold_tgt_id)

        for (test_src, test_target) in testalign:
            if (test_src, test_target) == ((), ()):
                continue
            if (test_src, test_target) in goldalign:
                # strict match
                tpstrict += 1
                tplax += 1
            else:
                # For anything with partial gold/test overlap on the source,
                #   see if there is also partial overlap on the gold/test target
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


    def score_multiple(self, gold_list, test_list, value_for_div_by_0=0.0):
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

        result = dict(recall_strict=rstrict,
                    recall_lax=rlax,
                    precision_strict=pstrict,
                    precision_lax=plax,
                    f1_strict=fstrict,
                    f1_lax=flax)

        return result

    def main(self):

        data_dir = self.results_dir
        avg_f1_novalign = 0
        avg_f1_vecalign = 0
        for filename in os.listdir(data_dir):
            gt_tid2sid = dict()
            novalign_tid2sid = dict()
            vecalign_tid2sid= dict()
            with open(data_dir + "/" + filename, "r") as fin:
                for i, line in enumerate(fin):
                    if i > 0 and line[0] != "":
                        line = line.strip().split("\t")
                        sid = line[0]

                        gt_target_ids = tuple(sorted(ast.literal_eval(line[2])))
                        novalign_target_ids = tuple(sorted(ast.literal_eval(line[4])))
                        vecalign_target_ids = tuple(sorted(ast.literal_eval(line[6])))

                        if gt_target_ids not in gt_tid2sid:
                            gt_tid2sid[gt_target_ids] = [sid]
                        else:
                            gt_tid2sid[gt_target_ids].append(sid)
                        if novalign_target_ids not in novalign_tid2sid:
                            novalign_tid2sid[novalign_target_ids] = [sid]
                        else:
                            novalign_tid2sid[novalign_target_ids].append(sid)
                        if vecalign_target_ids not in vecalign_tid2sid:
                            vecalign_tid2sid[vecalign_target_ids] = [sid]
                        else:
                            vecalign_tid2sid[vecalign_target_ids].append(sid)

                gt_sid2tid = {tuple(sorted(v)): k for k, v in gt_tid2sid.items() if len(k) > 0}
                for k, v in gt_tid2sid.items():
                    if len(k) == 0:
                        for sid in v:
                            gt_sid2tid[tuple([sid])] = tuple([])
                novalign_sid2tid = {tuple(sorted(v)): k for k, v in novalign_tid2sid.items() if len(k) > 0 and k != tuple([''])}
                for k, v in novalign_tid2sid.items():
                    if len(k) == 0 or k == tuple(['']):
                        for sid in v:
                            novalign_sid2tid[tuple([sid])] = tuple([])
                vecalign_sid2tid = {tuple(sorted(v)): k for k, v in vecalign_tid2sid.items() if len(k) > 0 and k != tuple([''])}
                for k, v in vecalign_tid2sid.items():
                    if len(k) == 0 or k == tuple(['']):
                        for sid in v:
                            vecalign_sid2tid[tuple([sid])] = tuple([])
            results_novalign = self.score_multiple([gt_sid2tid], [novalign_sid2tid])
            results_vecalign = self.score_multiple([gt_sid2tid], [vecalign_sid2tid])
            avg_f1_novalign += results_novalign["f1_strict"]
            avg_f1_vecalign += results_vecalign["f1_strict"]
            print(filename, results_novalign["f1_strict"], results_vecalign["f1_strict"])
        avg_f1_novalign = avg_f1_novalign / len(os.listdir(data_dir))
        avg_f1_vecalign = avg_f1_vecalign / len(os.listdir(data_dir))
        print(avg_f1_novalign, avg_f1_vecalign)

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("results_dir", type=str)
    args = parser.parse_args()
    paper_results = PaperResults(args.results_dir)
    paper_results.main()