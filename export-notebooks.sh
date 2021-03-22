#!/bin/bash

DIR_NOTEBOOKS="notebooks"
FORMAT="markdown"

NBVIEWER_LINK="https://nbviewer.jupyter.org/github/rueedlinger/machine-learning-snippets/blob/master"

quit() {
    echo "quit export notebook script!"
    exit 1
}


trap quit SIGINT
trap quit SIGTERM

function cleanup_md() {
    ######
    # Clenaup. First remove exsting markdonw files
    ######
    files=$(find $DIR_NOTEBOOKS -not -path '*/\.*' -name "*.md")

    for path in $files; do # Not recommended, will break on whitespace
        rm $path
    done
}

function remove_style() {
    ######
    # Remove HTML style tag in markdown
    ######
    files=$(find $DIR_NOTEBOOKS -not -path '*/\.*' -name "*.md")
    
    for path in $files; do # Not recommended, will break on whitespace
        sed -i '' '/<style.*>/,/<\/style>/d' $path
    done

}

function cleanup_images() {
    ######
    # Clenaup. First remove exsting png files
    ######
    files=$(find $DIR_NOTEBOOKS -not -path '*/\.*' -name "*.png")
    #
    for path in $files; do # Not recommended, will break on whitespace
        rm $path
    done
}

function export_notebooks() {
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
        echo -e ">**Note**: This is a generated markdown export from the Jupyter notebook file [$file]($file).\n>You can also view the notebook with the [nbviewer]($NBVIEWER_LINK/$path) from Jupyter. \n\n$(cat $newFile)" > $newFile
    done

}

function execute_notebooks() {
    ######
    # Execute notebooks
    ######
    files=$(find $DIR_NOTEBOOKS -not -path '*/\.*' -name "*.ipynb")

    for path in $files; do # Not recommended, will break on whitespace
        echo "execute notebook ${path}"
        # this will execute the notebooks
        pipenv run jupyter nbconvert --to notebook --inplace --execute $path 
        
    done

}

while test $# -gt 0
do
    case "$1" in
        --rm-md) 
            echo "argument $1"
            cleanup_md
            exit 0
            ;;
        --rm-img) 
            echo "argument $1"
            cleanup_images
            exit 0
            ;;
        --rm-style) 
            echo "argument $1"
            remove_style
            exit 0
            ;;
        --export) 
            echo "argument $1"
            export_notebooks
            exit 0
            ;;
        --run) 
            echo "argument $1"
            execute_notebooks
            exit 0
            ;;
        --*) 
            echo "bad option $1"
            exit 0
            ;;
        *)  
            echo "bad option $1"
            exit 0
            ;;
    esac
    shift
done

echo "export notebooks to markdown"
cleanup_md
cleanup_images
export_notebooks
remove_style






