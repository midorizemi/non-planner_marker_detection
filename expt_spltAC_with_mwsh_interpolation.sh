#!/usr/bin/env bash

# ------------------------------
# expt_split_affinesim_combinable_interporation.py
# を実行する実験用スクリプト
# pythonの環境はrequirement.txtを参照し，実行環境を整える．
# ------------------------------
export PYTHONPATH="$(pwd)/commons:$(pwd)/make_database:${PYTHONPATH}"
export PYTHON_PATH="$(pwd)/commons:$(pwd)/make_database:${PYTHON_PATH}"
# ------------------------------
# Python activate
# ------------------------------
source venv/bin/activate

# ------------------------------
# 実験用データ・セット周りのパス変数の設定
# ------------------------------
PROCJECT_DIR=`dirname $(pwd)`
echo $PROJECT_DIR
DATA_DIR=$PROCJECT_DIR"/data"
TEMPLATE_DIR=$DATA_DIR"/templates"
INPUTS_DIR=$DATA_DIR"/inputs"
OUTPUTS_DIR=$DATA_DIR"/outputs"

declare -a array=("qrmarker" "nabe")
#declare -a array=("glass" "ornament" "menko")
declare -a prefix=("pl_" "mltf_" "crv_")

data_flag="raw"
#data_flag="real"
ext=".png"

offset=57

for marker in ${array[@]}
do
    template_full_path=$TEMPLATE_DIR"/"$marker$ext
    echo "テンプレート>>"$template_full_path
    for prf in ${prefix[@]}
    do
        testset=${INPUTS_DIR}"/"${data_flag}"/"${prf}${marker}
        declare -a inputs=(`find ${testset} -type f |grep -E ".*[0-9]{3}_[0-9]{3}-[0-9]{3}.png" | sort`)
        COUNT=0
        for input_file in ${inputs[@]}
        do
            if [ $((COUNT % offset)) -ne 0 ]; then
                COUNT=$((++COUNT))
                continue 1
            fi
            echo "テスト画像>>"$input_file

            #op=$(python expt_split_affinesim.py "${template_full_path} ${input_file} ${prf} ${testset} --feature sift" | tail -n 1 >&1)
            #exec python "expt_split_affinesim.py" "${template_full_path}" $"{input_file}" "${prf}" "${testset}" "--feature sift"

            python expt_split_affinesim_combinable_interporation.py --feature=sift $template_full_path $input_file $prf $testset
            #echo "result>>> "$op
            COUNT=$((++COUNT))
            break
        done
#        break
    done
#    break
done
