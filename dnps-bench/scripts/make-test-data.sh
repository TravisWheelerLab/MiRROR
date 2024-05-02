#!/bin/bash
DATADIR="$1"
TESTDATADIR="$2"
NSAMPLES=$3
mkdir "$TESTDATADIR"
echo "$TESTDATADIR/"
for DATASET in "$DATADIR"/*; do
    PXDNAME=`basename $DATASET`
    TESTDATASET="$TESTDATADIR"/"$PXDNAME"
    mkdir "$TESTDATASET"
    echo "$TESTDATASET"
    COUNT=0
    for DATAFILE in "$DATASET"/*.raw; do
        FILENAME=`basename $DATAFILE`
        TESTDATAFILE="$TESTDATASET"/"$FILENAME"
        cp "$DATAFILE" "$TESTDATASET"
        echo `ls $TESTDATAFILE`
        let "COUNT+=1"
	echo "$COUNT"
	if [ $COUNT -ge $NSAMPLES ]; then
	    COUNT=0
            break
        fi
    done
    COUNT=0
    for DATAFILE in "$DATASET"/*.wiff; do
        FILENAME=`basename $DATAFILE`
        TESTDATAFILE="$TESTDATASET"/"$FILENAME"
        cp $DATAFILE* "$TESTDATASET"
	echo `ls $TESTDATAFILE*`
        let "COUNT+=1"
	echo "$COUNT"
	if [ $COUNT -ge $NSAMPLES ]; then
	    COUNT=0
            break
        fi
    done
done
