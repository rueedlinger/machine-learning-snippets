#!/bin/bash

######
# This script will export all the Jupyter Notebooks to Markdown.
######

DIR_NOTEBOOKS="notebooks"
FORMAT="markdown"

files=$(find $DIR_NOTEBOOKS -not -path '*/\.*' -name "*.ipynb")

for path in $files; do # Not recommended, will break on whitespace
    dir=$(dirname $path)
    file=$(basename $path)

    pipenv run jupyter nbconvert --to markdown $path
    newFile=${path//.ipynb/.md} 

    # add note to every markdown export
    echo -e ">**Note**: This is a generated markdown export from the Jupyter notebook file [$file]($file).\n\n$(cat $newFile)" > $newFile

done
