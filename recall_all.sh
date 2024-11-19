node="node09"

categories=(
    "Arts_Crafts_and_Sewing AmazonReviews2023/dataset/filtered_Arts_Crafts_and_Sewing.pkl 4096"
    "Automotive AmazonReviews2023/dataset/filtered_Automotive.pkl 8192"
    "Baby_Products AmazonReviews2023/dataset/filtered_Baby_Products.pkl 4096"
    "Cell_Phones_and_Accessories AmazonReviews2023/dataset/filtered_Cell_Phones_and_Accessories.pkl 4096"
    "Industrial_and_Scientific AmazonReviews2023/dataset/filtered_Industrial_and_Scientific.pkl 1024"
    "Musical_Instruments AmazonReviews2023/dataset/filtered_Musical_Instruments.pkl 1024"
    "Office_Products AmazonReviews2023/dataset/filtered_Office_Products.pkl 4096"
    "Patio_Lawn_and_Garden AmazonReviews2023/dataset/filtered_Patio_Lawn_and_Garden.pkl 4096"
    "Sports_and_Outdoors AmazonReviews2023/dataset/filtered_Sports_and_Outdoors.pkl 4096"
    "Tools_and_Home_Improvement AmazonReviews2023/dataset/filtered_Tools_and_Home_Improvement.pkl 16384"
    "OnlineRetail OnlineRetail/filtered_OnlineRetail.pkl 1024"
    "HM HM/filtered_HM.pkl 16384"
    "Beauty_and_Personal_Care AmazonReviews2023/dataset/filtered_Beauty_and_Personal_Care.pkl 8192"
    "Georgia_Fashion GoogleLocalData/Fashion/rating_dataset/filtered_Georgia.pkl 1024"
    "Illinois_Fashion GoogleLocalData/Fashion/rating_dataset/filtered_Illinois.pkl 1024"
    "Michigan_Fashion GoogleLocalData/Fashion/rating_dataset/filtered_Michigan.pkl 1024"
    "New_York_Fashion GoogleLocalData/Fashion/rating_dataset/filtered_New_York.pkl 1024"
    "Ohio_Fashion GoogleLocalData/Fashion/rating_dataset/filtered_Ohio.pkl 1024"
    "Pennsylvania_Fashion GoogleLocalData/Fashion/rating_dataset/filtered_Pennsylvania.pkl 1024"
    "Goodreads Goodreads/filtered_Goodreads_rating.pkl 16384"
    "Books AmazonReviews2023/dataset/filtered_Books.pkl 16384"
    "MovieLens MovieLens/filtered_MovieLens.pkl 16384"
    "Toys_and_Games AmazonReviews2023/dataset/filtered_Toys_and_Games.pkl 8192"
    "Video_Games AmazonReviews2023/dataset/filtered_Video_Games.pkl 1024"
    "Bili_Cartoon Bili/dataset/filtered_Bili_Cartoon.pkl 512"
    "Bili_Dance Bili/dataset/filtered_Bili_Dance.pkl 512"
    "Bili_Food Bili/dataset/filtered_Bili_Food.pkl 512"
    "Bili_Movie Bili/dataset/filtered_Bili_Movie.pkl 512"
    "Bili_Music Bili/dataset/filtered_Bili_Music.pkl 512"
    "DY Bili/dataset/filtered_DY.pkl 512"
    "KU Bili/dataset/filtered_KU.pkl 512"
    "QB Bili/dataset/filtered_QB.pkl 512"
    # "TN Bili/dataset/filtered_TN.pkl 512"
    "Yelp Yelp/filtered_Yelp_rating.pkl 2048"
    "Arkansas_Food GoogleLocalData/Food/rating_dataset/filtered_Arkansas.pkl 1024"
    "Delaware_Food GoogleLocalData/Food/rating_dataset/filtered_Delaware.pkl 1024"
    "District_of_Columbia_Food GoogleLocalData/Food/rating_dataset/filtered_District_of_Columbia.pkl 1024"
    "Hawaii_Food GoogleLocalData/Food/rating_dataset/filtered_Hawaii.pkl 1024"
    "Idaho_Food GoogleLocalData/Food/rating_dataset/filtered_Idaho.pkl 1024"
    "Maine_Food GoogleLocalData/Food/rating_dataset/filtered_Maine.pkl 1024"
)  

num_processes=16
cudas="0,1,2,3,4,5,6,7"
cuda_array=(${cudas//,/ })

run_parallel() {
    cuda_id="$1"
    c="$2"
    echo $cuda_id $c &
    bash run_recall_one $1 $2
    # sbatch sbatch_run_recall_one.sh $1 $2
    sleep 1m
}

index=0
for c in "${categories[@]}"; do
    while [ $(squeue -u lyw | grep $node | wc -l) -ge $num_processes ]; do
        sleep 5m
    done

    if ((index >= ${#cuda_array[@]})); then
        index=0
    fi

    cuda_id="${cuda_array[index]}"
    run_parallel "$cuda_id" "$c"

    ((index += 1))
done
