categories=(
    # "Source_datasets/Bili_2M"
    "Downstream_datasets/Bili_Cartoon"
    "Downstream_datasets/Bili_Dance"
    "Downstream_datasets/Bili_Food"
    "Downstream_datasets/Bili_Movie"
    "Downstream_datasets/Bili_Music"
    "Downstream_datasets/DY"
    "Downstream_datasets/KU"
    "Downstream_datasets/TN"
    "Downstream_datasets/QB"
)


for name in "${categories[@]}"; do
    python "$1.py" $name
    echo "${name} has been processed."
done