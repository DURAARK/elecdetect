#!/bin/bash

configs=(
"-1 gof -f rf"
)
DIR_build="Release"
DIR_trainingset="resource/trainingset_128"
DIR_testset="resource/testset"

#this is hardcoded, do not change
NAME_result="results" 
postprocess_images=("original_labels.jpg" "closed_labels.jpg" "prop_map.jpg" "detect_result.jpg")

echo "Executing Testing script in testset $DIR_testset..."

for cur_config in "${configs[@]}"
do
    echo ""
    echo "executing: $cur_config at "
    date
    echo "--------------------------"
    if [ ! -e "./$DIR_testset/$NAME_result $cur_config/config.xml" ]
    then
       echo "\"./$DIR_testset/$NAME_result $cur_config/config.xml\" doesn't exist.. so train.."
        ./$DIR_build/ElecDetec.a $cur_config -d ./$DIR_trainingset -c temp_config.xml
    else
       echo "\"./$DIR_testset/$NAME_result $cur_config/config.xml\" exists! Skipping training."
       cp "./$DIR_testset/$NAME_result $cur_config/config.xml" "temp_config.xml" 
    fi

    echo ""
    echo "Starting Testing. Its "
    date
    echo ""

    ./$DIR_build/ElecDetec.a -c temp_config.xml -d ./$DIR_testset
    rm "./$DIR_testset/$NAME_result $cur_config" -f -r
    mv "./$DIR_testset/$NAME_result" "./$DIR_testset/$NAME_result $cur_config"
    mv "temp_config.xml" "./$DIR_testset/$NAME_result $cur_config/config.xml"
    
    for image in "${postprocess_images[@]}"
    do
       mv $image "./$DIR_testset/$NAME_result $cur_config/$image"
    done

    echo ""
    echo "Config done. Its "
    date
    echo ""
done







