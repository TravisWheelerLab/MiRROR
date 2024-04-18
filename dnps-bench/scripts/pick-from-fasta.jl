using DataFrames, CSV, FASTX

function readfasta(path)
    collect(
        FASTX.FASTA.Reader(
            open(path)))
end

function writefasta(path,records)
    FASTX.FASTA.Writer(open(path,"w")) do fastafile
        for record = records
            write(fastafile,record)
        end
    end
end

function filter_fasta_by_description(
    out_path::AbstractString,
    description_keys::Vector{<:AbstractString},
    assembly_paths::Vector{<:AbstractString}
)
    assemblies_union = reduce(vcat,readfasta.(assembly_paths))
    # selected_records = filter(x -> any(contains(description(x),key) for key in description_keys), assemblies_union)
    selected_records = []
    match_counts = Int[]
    for key in description_keys
        count = 0
        for record in assemblies_union
            if contains(description(record),key)
                push!(selected_records,record)
                count += 1
            end
        end
        push!(match_counts,count)
    end
    print(sum(match_counts) / length(description_keys))
    writefasta(out_path,selected_records)
end