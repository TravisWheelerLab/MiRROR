for line in `cat $1`
do
	./scripts/wget-ftp-pride.sh $line $2
done
rm '*/index*'
