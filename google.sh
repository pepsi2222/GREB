#!/bin/bash

# 定义类别列表
categories=(
    "Alabama"
    "Alaska"
    "Arizona"
    "Arkansas"
    "California"
    "Colorado"
    "Connecticut"
    "Delaware"
    "District_of_Columbia"
    "Florida"
    # "Georgia"
    # "Hawaii"
    # "Idaho"
    # "Illinois"
    # "Indiana"
    # "Iowa"
    # "Kansas"
    # "Kentucky"
    # "Louisiana"
    # "Maine"
    # "Maryland"
    # "Massachusetts"
    # "Michigan"
    # "Minnesota"
    # "Mississippi"
    # "Missouri"
    # "Montana"
    # "Nebraska"
    # "Nevada"
    # "New_Hampshire"
    # "New_Jersey"
    # "New_Mexico"
    # "New_York"
    # "North_Carolina"
    # "North_Dakota"
    # "Ohio"
    # "Oklahoma"
    # "Oregon"
    # "Pennsylvania"
    # "Rhode_Island"
    # "South_Carolina"
    # "South_Dakota"
    # "Tennessee"
    # "Texas"
    # "Utah"
    # "Vermont"
    # "Virginia"
    # "Washington"
    # "West_Virginia"
    # "Wisconsin"
    # "Wyoming"
)

# 定义CUDA设备组
devices=("0,1" "2,5" "4,3" "6,7")

# 当前正在运行的任务数
current_jobs=0

# 启动一个任务的函数
start_task() {
    local category=$1
    local device=$2

    echo "Starting task for category ${category} on devices ${device}"
    CUDA_VISIBLE_DEVICES="${device}" python script/run_ctr.py "${category}" &
}

# 启动任务的主循环
for category in "${categories[@]}"; do
    while [ "${current_jobs}" -ge 8 ]; do
        # 等待任何一个任务完成
        wait -n
        current_jobs=$((current_jobs - 1))
    done

    # 选择下一个可用设备组
    device=${devices[$((current_jobs / 2 % 4))]}
    
    start_task "${category}" "${device}"
    current_jobs=$((current_jobs + 1))
done

# 等待所有任务完成
wait

echo "All tasks have been trained."
