#!/bin/bash
DATADIR="$1"
PXDNAME=""
FILENAME=""
COUNT=0

convert_to_mzMLb(){
    INPUTPATH="data/$PXDNAME/$FILENAME"
    OUTPUTDIR="data/$PXDNAME"
    echo number "$COUNT"
    echo inputpath "$INPUTPATH"
    echo outputpath "$OUTPUTPATH"
    singularity exec --writable --cleanenv -B "$DATADIR":/data -B `mktemp -d /home/u24/georgeglidden/pwiz-msconvert/wineXXX`:/mywineprefix mywine msconvert "$INPUTPATH" -o "$OUTPUTDIR" --mzMLb
}

for DATASET in "$DATADIR/PXD*"; do
    PXDNAME=`basename $DATASET`
    for DATAFILE in "$DATASET/*.raw"; do
        FILENAME=`basename $DATAFILE`
        convert_to_mzMLb() &
        let "COUNT+=1"
    done
    for DATAFILE in "$DATASET/*.wiff"; do
        FILENAME=`basename $DATAFILE`
        convert_to_mzMLb() &
        let "COUNT+=1"
    done
done