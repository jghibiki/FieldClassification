#!/bin/bash

a=0
p="../../images/"


for d in ./*/; do
    cd "$d";
    ((a++));

    
    f1=$p$a
    f1+="_image.png"
    cp image.png $f1;

    f2=$p$a
    f2+="_label.png"
    cp label.png $f2;
   
    echo "$a";
    cd ..

done;
