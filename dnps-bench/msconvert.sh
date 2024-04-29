PWIZDIR=/home/u24/georgeglidden/pwiz-msconvert/pwiz_sandbox
DATADIR=/xdisk/twheeler/georgeglidden/pivotpoint/dnps-bench/data
DATASET="$1"
DATAFILE="$2"

singularity exec --cleanenv \
	-B "$DATADIR":/data \
	-B `mktemp -d /dev/shm/wineXXX`:/mywineprefix \
	--writable-tmpfs "$PWIZDIR" \
	mywine msconvert /data/"$DATASET"/"$DATAFILE" -o /data/"$DATASET"/

