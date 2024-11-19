dataset=(
    # "/data1/home/xingmei/GRE/data/AmazonReviews2023/dataset/filtered_CDs_and_Vinyl.pkl"
    "/data1/home/xingmei/GRE/data/AmazonReviews2023/dataset/filtered_Electronics.pkl"
    # "/data1/home/xingmei/GRE/data/AmazonReviews2023/dataset/filtered_Health_and_Household.pkl"
    # "/data1/home/xingmei/GRE/data/AmazonReviews2023/dataset/filtered_Home_and_Kitchen.pkl"
    # "/data1/home/xingmei/GRE/data/AmazonReviews2023/dataset/filtered_Pet_Supplies.pkl"
    # "/data1/home/xingmei/GRE/data/AmazonReviews2023/dataset/filtered_Software.pkl"

    # "/data1/home/xingmei/GRE/data/AmazonReviews2023/dataset/filtered_Clothing_Shoes_and_Jewelry.pkl"
    # "/data1/home/xingmei/GRE/data/GoogleLocalData/Fashion/rating_dataset/filtered_California.pkl"
    # "/data1/home/xingmei/GRE/data/GoogleLocalData/Fashion/rating_dataset/filtered_Florida.pkl"
    # "/data1/home/xingmei/GRE/data/GoogleLocalData/Fashion/rating_dataset/filtered_Texas.pkl"

    # "/data1/home/xingmei/GRE/data/AmazonReviews2023/dataset/filtered_Kindle_Store.pkl"
    # "/data1/home/xingmei/GRE/data/Steam/filtered_Steam.pkl"
    # "/data1/home/xingmei/GRE/data/Bili/dataset/filtered_Bili_2M.pkl"

    # "/data1/home/xingmei/GRE/data/GoogleLocalData/Food/rating_dataset/filtered_Alabama.pkl"
    # "/data1/home/xingmei/GRE/data/GoogleLocalData/Food/rating_dataset/filtered_Connecticut.pkl"
    # "/data1/home/xingmei/GRE/data/GoogleLocalData/Food/rating_dataset/filtered_Mississippi.pkl"
    # "/data1/home/xingmei/GRE/data/AmazonReviews2023/dataset/filtered_Grocery_and_Gourmet_Food.pkl"
)

config=(
    # "/data1/home/xingmei/GRE/data/AmazonReviews2023/config/CDs_and_Vinyl.yaml"
    "/data1/home/xingmei/GRE/data/AmazonReviews2023/config/Electronics.yaml"
    # "/data1/home/xingmei/GRE/data/AmazonReviews2023/config/Health_and_Household.yaml"
    # "/data1/home/xingmei/GRE/data/AmazonReviews2023/config/Pet_Supplies.yaml"
    # "/data1/home/xingmei/GRE/data/AmazonReviews2023/config/Software.yaml"

    # "/data1/home/xingmei/GRE/data/AmazonReviews2023/config/Clothing_Shoes_and_Jewelry.yaml"
    # "/data1/home/xingmei/GRE/data/GoogleLocalData/Fashion/config/California.yaml"
    # "/data1/home/xingmei/GRE/data/GoogleLocalData/Fashion/config/Florida.yaml"
    # "/data1/home/xingmei/GRE/data/GoogleLocalData/Fashion/config/Texas.yaml"

    # "/data1/home/xingmei/GRE/data/AmazonReviews2023/config/Kindle_Store.yaml"
    # "/data1/home/xingmei/GRE/data/Steam/Steam.yaml"
    # "/data1/home/xingmei/GRE/data/Bili/config/Bili_2M.yaml"

    # "/data1/home/xingmei/GRE/data/GoogleLocalData/Food/config/Alabama.yaml"
    # "/data1/home/xingmei/GRE/data/GoogleLocalData/Food/config/Connecticut.yaml"
    # "/data1/home/xingmei/GRE/data/GoogleLocalData/Food/config/Mississippi.yaml"
    # "/data1/home/xingmei/GRE/data/AmazonReviews2023/config/Grocery_and_Gourmet_Food.yaml"
)

# python gre.py \
#     --model TE_SASRec \
#     --dataset_path /data1/home/xingmei/GRE/data/AmazonReviews2023/dataset/filtered_CDs_and_Vinyl.pkl \
#     --data_config_path /data1/home/xingmei/GRE/data/AmazonReviews2023/config/CDs_and_Vinyl.yaml \
#     --gpu 1 \
#     --embed_dim 64 \
#     --batch_size 2048

length=${#dataset[@]}
for (( i=0; i<$length; i++ )); do
    ckpt=$(<"transrec_ckpt.txt")
    python gre.py \
        --model TE_SASRec \
        --dataset_path ${dataset[$i]} \
        --data_config_path  ${config[$i]} \
        --gpu 1
        # --embed_dim 64 \
        # --batch_size 2048 \
        # --ckpt_path $ckpt
done