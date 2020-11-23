#!/bin/bash

DIR_NOTEBOOKS="notebooks"
FORMAT="markdown"

######
# Clenaup. First remove exsting markdonw files
######
# files=$(find $DIR_NOTEBOOKS -not -path '*/\.*' -name "*.md")
#
#for path in $files; do # Not recommended, will break on whitespace
#    rm $path
#done


######
# Clenaup. First remove exsting png files
######
#files=$(find $DIR_NOTEBOOKS -not -path '*/\.*' -name "*.png")
#
#for path in $files; do # Not recommended, will break on whitespace
#    rm $path
#done

######
# Export notebooks to markdown
######
files=$(find $DIR_NOTEBOOKS -not -path '*/\.*' -name "*.ipynb")

for path in $files; do # Not recommended, will break on whitespace
    dir=$(dirname $path)
    file=$(basename $path)

    # this will execute the notebooks first and the export the result as markdonw
    pipenv run jupyter nbconvert --execute --to markdown $path
    newFile=${path//.ipynb/.md} 

    # add note to every markdown export
    echo -e ">**Note**: This is a generated markdown export from the Jupyter notebook file [$file]($file).\n\n$(cat $newFile)" > $newFile

done
