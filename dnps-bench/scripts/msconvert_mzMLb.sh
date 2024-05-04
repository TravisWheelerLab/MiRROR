#!/bin/sh
PWIZDIR=/home/u24/georgeglidden/pwiz-msconvert/pwiz_sandbox

DATASET="$1"
DATAFILE="$2"
DATADIR="$3"

INPUTPATH=/data/"$DATASET"/"$DATAFILE"
OUTPUTDIR=/data/"$DATASET"/

singularity exec --writable --cleanenv -B "$DATADIR":/data -B `mktemp -d /home/u24/georgeglidden/pwiz-msconvert/wineXXX`:/mywineprefix --writable-tmpfs "$PWIZDIR" mywine msconvert "$INPUTPATH" -o "$OUTPUTDIR" --mzMLb \
    > "$DATADIR"/"$DATASET"/"$DATAFILE"_msconvert_out