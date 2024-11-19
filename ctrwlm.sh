conda activate rec
CUDA_VISIBLE_DEVICES="$1,$2" python script/run_ctr_with_text.py $3