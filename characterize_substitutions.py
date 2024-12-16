from run_test import *
import mirror
import sys
AMINOS = mirror.util.AMINOS
mask_res = mirror.util.mask_ambiguous_residues
if __name__ == "__main__":
    fasta_path = sys.argv[1]
    seqs = mirror.io.load_fasta_as_strings(fasta_path)
    candidates, misses, crashes = run_on_seqs(seqs, None, None, verbose = False)
    substitutions = { mask_res(res): { mask_res(res2): 0 for res2 in AMINOS} for res in AMINOS}
    for data in misses:
        if len(data.candidates) == 0:
            continue
        else:
            for _, __, cand in data.candidates:
                optimum, optimizer = cand.edit_distance(data.peptide)
                if optimum == 1:
                    target = candidate.construct_target(data.peptide)
                    best_query = cand._sequences[optimizer]
                    if len(target) == len(best_query):
                        for i in range(len(target)):
                            target_chr = target[i]
                            query_chr = best_query[i]
                            if target_chr != query_chr:
                                substitutions[target_chr][query_chr] += 1
    for target_chr in AMINOS:
        target_chr = mask_res(target_chr)
        for query_chr in AMINOS:
            query_chr = mask_res(query_chr)
            num_subs = substitutions[target_chr][query_chr]
            if num_subs > 0:
                print(f"{target_chr} => {query_chr} : {num_subs}")

                    