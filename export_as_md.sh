#!/bin/bash

FORMAT=markdown
OUT=out

rm -r $OUT

mkdir $OUT

find ./unsupervised -not -path '*/\.*' -name '*[a-zA-Z].ipynb' | while read line; do
    #echo $line
    jupyter nbconvert --to $FORMAT $line 
    DATE=`stat -f "%Sm" -t "%Y-%m-%d" $line`
    
    
    md="${line/.ipynb/.md}"
    TITLE=`cat $md | grep '# ' | sed 's/#//g'`
    
    CONTENT=`sed 's/#.*//g' $md`
    
    echo -e "---\ntitle: $TITLE\ndate: $DATE\n---\n $CONTENT" > $md
    mv $md $OUT
    mv "${line/.ipynb/_files}" $OUT
done
