for line in `cat $1`
do
	./scripts/wget-pride-index.sh $line
done
julia ./scripts/parse-filesizes-from-index.jl "$PWD"
rm index*