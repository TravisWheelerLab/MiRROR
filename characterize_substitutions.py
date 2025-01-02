from run_test import *
import mirror
import sys
AMINOS = mirror.util.AMINOS
mask_res = mirror.util.mask_ambiguous_residues
if __name__ == "__main__":
    # read the data, re-run analysis
    fasta_path = sys.argv[1]
    seqs = mirror.io.load_fasta_as_strings(fasta_path)
    candidates, misses, crashes = run_on_seqs(seqs, None, None, verbose = False)
    
    # accumulate single-character substitutions
    substitutions = { mask_res(res): { mask_res(res2): 0 for res2 in AMINOS} for res in AMINOS}
    column_total = { mask_res(res) : 0 for res in AMINOS}
    row_total = { mask_res(res) : 0 for res in AMINOS}
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
                                column_total[target_chr] += 1
                                row_total[query_chr] += 1

    # display substitution type counts
    substitution_types = []
    for target_chr in AMINOS:
        target_chr = mask_res(target_chr)
        for query_chr in AMINOS:
            query_chr = mask_res(query_chr)
            num_subs = substitutions[target_chr][query_chr]
            if num_subs > 0:
                substitution_types.append((target_chr, query_chr, num_subs))
    substitution_types.sort(key = lambda x: x[2])
    substitution_types_str = '\n'.join(map(
        lambda x: f"{x[0]} => {x[1]}: \t{x[2]}", 
        substitution_types))
    print(f"occurrence of substitution types:\n{substitution_types_str}")

    # display marginal counts for source
    column_total_str = '\n'.join(map(
        lambda x: f"{x[0]}:\t{x[1]}", 
        sorted(column_total.items(), key = lambda x: x[1])))
    print(f"marginal occurrence of substitution source:\n{column_total_str}")

    # display marginal counts for target
    row_total_str = '\n'.join(map(
        lambda x: f"{x[0]}:\t{x[1]}", 
        sorted(row_total.items(), key = lambda x: x[1])))
    print(f"marginal occurrence of substitution target:\n{row_total_str}")