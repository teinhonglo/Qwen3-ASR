
#!/bin/bash
# dependency: torch, torchaudio, transformers, datasets, librosa

set -euo pipefail

stage=0
stop_stage=1000
data_root=data
test_sets="mac_slu_train mac_slu_dev mac_slu_test"
max_new_tokens=256
model_path="Qwen/Qwen3-ASR-1.7B"
gpuid=0
language=Chinese

. ./local/parse_options.sh
. ./path.sh

output_root=exp/$(basename $model_path | tr 'A-Z' 'a-z')

if [ $stage -le 1 ] && [ $stop_stage -ge 1 ]; then
    for test_set in $test_sets; do
        data_dir=$data_root/$test_set
        output_dir=$output_root/$test_set
        echo "data_root: $data_root and output_dir: $output_dir"
        
        CUDA_VISIBLE_DEVICES=$gpuid \
            python demo/qwen3_asr_inference_from_datadir.py \
                --language $language \
                --model_path $model_path \
                --max_new_tokens $max_new_tokens \
                --data_dir $data_dir \
                --output_dir $output_dir
    done
fi

if [ $stage -le 2 ] && [ $stop_stage -ge 2 ]; then
    dest_dir="/share/nas167/teinhonglo/espnets/espnet-hakka/egs2/l2arctic/asr1/exp/stt"

    output_name=$(basename $output_root)
    if [ -d $dest_dir/$output_name ]; then
        rm -rf $dest_dir/$output_name
    fi

    rsync -avP exp/$output_name $dest_dir/
    cd /share/nas167/teinhonglo/espnets/espnet-hakka/egs2/l2arctic/asr1
    ./run_stt.sh --test_sets "$test_sets" --whisper_tag $output_name --stage 2 --stop-stage 2
fi


