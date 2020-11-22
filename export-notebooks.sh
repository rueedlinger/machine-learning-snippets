#!/bin/bash

######
# This script will export all the Jupyter Notebooks to a format like mardkowns, etc.
######

DIR_NOTEBOOKS="notebooks"
DIR_OUTPUT="out"
FORMAT="markdown"

files=$(find $DIR_NOTEBOOKS -not -path '*/\.*' -name "*.ipynb")

for path in $files; do # Not recommended, will break on whitespace
    dir=$(dirname $path)
    #file=$(basename $path)

    newOutputDir=${dir//$DIR_NOTEBOOKS/$DIR_OUTPUT}
    mkdir -p $newOutputDir

    pipenv run jupyter nbconvert --to markdown --output-dir=$newOutputDir $path

done
