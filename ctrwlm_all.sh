categories=(
    "Arts_Crafts_and_Sewing"
    "Automotive"
    "Baby_Products"
    "Cell_Phones_and_Accessories"
    "Industrial_and_Scientific"
    "Musical_Instruments"
    "Office_Products"
    "Patio_Lawn_and_Garden"
    "Sports_and_Outdoors"
    "Tools_and_Home_Improvement"

    "OnlineRetail"


    "Beauty_and_Personal_Care"


    "Goodreads"

    "Books"

    "Toys_and_Games"
    "Video_Games"

    "Bili_Cartoon"
    "Bili_Dance"
    "Bili_Food"
    "Bili_Movie"
    "Bili_Music"
    "DY"
    "KU"
    "QB"
    "TN"

    "Yelp"

    "Maine_Food"
)  

num_processes=8
python_script="script/run_ctr_with_text.py"
cudas="0,1,2,3,1,0,3,2"
cuda_array=(${cudas//,/ })

run_parallel() {
    cuda_id="$1"
    c="$2"
    # echo $cuda_id $c && sleep 10s &
    CUDA_VISIBLE_DEVICES=$cuda_id python $python_script $c &
}

index=0
for c in "${categories[@]}"; do
    # 限制并行进程的数量
    while [ $(jobs | wc -l) -ge $num_processes ]; do
        wait -n
    done

    # 确保index不会超过cuda_array的长度
    if ((index + 1 >= ${#cuda_array[@]})); then
        index=0
    fi

    # 拿两个CUDA ID
    cuda_ids="${cuda_array[index]},${cuda_array[index+1]}"
    run_parallel "$cuda_ids" "$c"
    
    # 每次增加2以确保每个任务都获取一对新的CUDA ID
    ((index += 2))
done
