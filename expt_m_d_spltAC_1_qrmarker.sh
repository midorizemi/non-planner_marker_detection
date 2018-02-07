#!/usr/bin/env bash

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
echo $DIR
export PYTHONPATH="${DIR}/makedb/commons:${DIR}/makedb/make_database:$PYTHONPATH"
export PYTHON_PATH="${DIR}/makedb/commons:${DIR}/makedb/make_database:$PYTHON_PATH"

# ------------------------------
#Aliases settings
# ------------------------------
#alias blender="$HOME/Sources/blender-2.78a/blender"
alias activate="source $PYENV_ROOT/versions/anaconda3-5.0.0/bin/activate"
alias deactivate="source $PYENV_ROOT/versions/anaconda3-5.0.0/bin/deactivate"

source activate py36cv3
#activate py36cv34

PROCJECT_DIR="$( cd "$( dirname "${DIR}" )" && pwd )"
echo $PROJECT_DIR
DATA_DIR=$PROCJECT_DIR"/data"
TEMPLATE_DIR=$DATA_DIR"/templates"
INPUTS_DIR=$DATA_DIR"/inputs"
OUTPUTS_DIR=$DATA_DIR"/outputs"

declare -a array=("qrmarker" "nabe")
#declare -a array=("glass" "ornament" "menko")
declare -a prefix=("pl_" "mltf_" "crv_")

data_flag="cgs"
#data_flag="real"
ext=".png"

for marker in ${array[@]}
do
    template_full_path=$TEMPLATE_DIR"/"$marker$ext
    echo $template_full_path
    for prf in ${prefix[@]}
    do
        testset=${INPUTS_DIR}"/"${data_flag}"/"${prf}${marker}
        declare -a inputs=(`find ${testset} -type f`)
        for input_file in ${inputs[@]}
        do
            echo $input_file

            #python expt_hoge.py $template_full_path $input_file $prf $testset
        done
    done
done