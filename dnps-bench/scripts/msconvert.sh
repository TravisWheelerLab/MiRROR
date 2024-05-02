#!/bin/sh
PWIZDIR=/home/u24/georgeglidden/pwiz-msconvert/pwiz_sandbox
DATADIR=/xdisk/twheeler/georgeglidden/pivotpoint/dnps-bench/data

DATASET="$1"
DATAFILE="$2"

INPUTPATH=/data/"$DATASET"/"$DATAFILE"
OUTPUTDIR=/data/"$DATASET"/

echo pwizdir "$PWIZDIR"
echo datadir "$DATADIR"
echo dataset "$DATASET"
echo datafile "$DATAFILE"
echo inputpath "$INPUTPATH"
echo outputdir "$OUTPUTDIR"

singularity exec --writable --cleanenv -B "$DATADIR":/data -B `mktemp -d /home/u24/georgeglidden/pwiz-msconvert/wineXXX`:/mywineprefix --writable-tmpfs "$PWIZDIR" mywine msconvert "$INPUTPATH" -o "$OUTPUTDIR"