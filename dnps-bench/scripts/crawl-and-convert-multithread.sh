#!/bin/bash
PWIZDIR=/home/u24/georgeglidden/pwiz-msconvert/pwiz_sandbox
PXDNAME=""
FILENAME=""
COUNT=0

DATADIR="$1"
MAXP=$2

pwait(){
    while [ $(jobs -p -r | wc -l) -ge $1 ]; do
        sleep 1
    done
}

try_convert(){
    DATADIR=$1
    PXDNAME=$2
    DATAFILE=$3
    if test -f "$DATAFILE"; then
        FILENAME=`basename $DATAFILE`
        date
        echo $COUNT $DATADIR $PXDNAME $FILENAME
        ./msconvert_mzMLb.sh "$PXDNAME" "$FILENAME" "$DATADIR" &
        ((COUNT++))
        pwait $MAXP
    fi
}

for DATASET in "$DATADIR"/PXD*; do
    PXDNAME=`basename $DATASET`
    
    for DATAFILE in "$DATASET"/*.raw; do
        try_convert $DATADIR $PXDNAME $DATAFILE
    done
    
    for DATAFILE in "$DATASET"/*.wiff; do
        try_convert $DATADIR $PXDNAME $DATAFILE
    done
done

wait
date