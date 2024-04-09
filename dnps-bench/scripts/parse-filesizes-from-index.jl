using HTMLForge

extract_filesize_string_from_row(row_elt) =
    row_elt.children[4].children[1].text

extract_table_rows_from_index_doc(pride_index_html_doc::HTMLDocument) =
    pride_index_html_doc.root.children[2].children[2][1].children[4:end - 1]

function parse_filesize_notation(abbreviated_filesize::AbstractString)
    if all(isnumeric.(collect(abbreviated_filesize)))
        return parse(Int,abbreviated_filesize)
    else
        scalar = parse(Float64,abbreviated_filesize[1:end - 1])
        magnitude_notation = abbreviated_filesize[end]
        if magnitude_notation == 'K'
            return scalar * 1e3
        elseif magnitude_notation == 'M'
            return scalar * 1e6
        elseif magnitude_notation == 'G'
            return scalar * 1e9
        else
            return scalar
        end
    end
end

function parse_file_sizes(pride_index_file::AbstractString)
    index_html_doc = parsehtml(read(pride_index_file,String))
    row_elts = extract_table_rows_from_index_doc(index_html_doc)
    row_filesize_strings = extract_filesize_string_from_row.(row_elts)
    parse_filesize_notation.(row_filesize_strings)
end
    
function main()
    workspace = readdir(ARGS[1])
    index_files = filter(x->contains(x,"index.html"),workspace)
    file_size_lists = parse_file_sizes.(index_files)
    file_size_sums = sum.(file_size_lists)
    file_size_sum_total = sum(file_size_sums)
    println(index_files)
    println(file_size_sums)
    println(file_size_sum_total)
end

main()