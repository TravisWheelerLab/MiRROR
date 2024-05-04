for FOLDER in "$@"; do
    echo dataset "$FOLDER"
    echo raw `ls -l "$FOLDER"/*.raw | wc -l`
    echo wiff `ls -l "$FOLDER"/*.wiff | wc -l`
    echo mzMLb `ls -l "$FOLDER"/*.mzMLb | wc -l`
    echo out `ls -l "$FOLDER"/*out | wc -l`
done
