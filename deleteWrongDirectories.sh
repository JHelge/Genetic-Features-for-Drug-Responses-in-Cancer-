find . -type d -name 'Drug*_analysis' -print | while read dir; do
    # Check if the 'run0' directory exists inside the found directory
    if [ ! -d "$dir/run0" ]; then
        echo "Deleting $dir as it does not contain run0"
        echo "$dir"
        #rm -rf "$dir"
    #else
        #echo "$dir contains run0, not deleting"
    fi
done
