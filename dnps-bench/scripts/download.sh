for line in `cat $1`
do
	./scripts/wget-ftp-pride.sh $line
done
rm '*/index*'
