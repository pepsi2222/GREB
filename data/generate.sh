#!/bin/bash

# template="{name}:
#     path: AmazonReviews2023/dataset/{name}.pkl
#     text_fields: [title, description, features]
#     max_behavior_len: 20
#     naive_ctr:
#         learning_rate: 3e-3
#         factor: 0.3
#         patience: 0
#         train:
#             output_dir: AmazonReviews2023/{name}/ctr
#             per_device_train_batch_size: 2048
#             per_device_eval_batch_size: 2048
#             logging_strategy: epoch
#             save_strategy: epoch
#             evaluation_strategy: epoch
# "

# categories=(
#     # "All_Beauty"
#     # "Amazon_Fashion"
#     # "Appliances"
#     # "Arts_Crafts_and_Sewing"
#     # "Automotive"
#     # "Baby_Products"
#     # "Beauty_and_Personal_Care"
#     # "Books"
#     # "CDs_and_Vinyl"
#     "Cell_Phones_and_Accessories"
#     "Clothing_Shoes_and_Jewelry"
#     "Digital_Music"
#     "Electronics"
#     "Gift_Cards"
#     "Grocery_and_Gourmet_Food"
#     "Handmade_Products"
#     "Health_and_Household"
#     "Health_and_Personal_Care"
#     "Home_and_Kitchen"
#     "Industrial_and_Scientific"
#     "Kindle_Store"
#     "Magazine_Subscriptions"
#     "Movies_and_TV"
#     "Musical_Instruments"
#     "Office_Products"
#     "Patio_Lawn_and_Garden"
#     "Pet_Supplies"
#     "Software"
#     "Sports_and_Outdoors"
#     "Subscription_Boxes"
#     "Tools_and_Home_Improvement"
#     "Toys_and_Games"
#     "Video_Games"
# )

# categories=(
#     "Alabama"
#     "Alaska"
#     "Arizona"
#     "Arkansas"
#     "California"
#     "Colorado"
#     "Connecticut"
#     "Delaware"
#     "District_of_Columbia"
#     "Florida"
#     "Georgia"
#     "Hawaii"
#     "Idaho"
#     "Illinois"
#     "Indiana"
#     "Iowa"
#     "Kansas"
#     "Kentucky"
#     "Louisiana"
#     "Maine"
#     "Maryland"
#     "Massachusetts"
#     "Michigan"
#     "Minnesota"
#     "Mississippi"
#     "Missouri"
#     "Montana"
#     "Nebraska"
#     "Nevada"
#     "New_Hampshire"
#     "New_Jersey"
#     "New_Mexico"
#     "New_York"
#     "North_Carolina"
#     "North_Dakota"
#     "Ohio"
#     "Oklahoma"
#     "Oregon"
#     "Pennsylvania"
#     "Rhode_Island"
#     "South_Carolina"
#     "South_Dakota"
#     "Tennessee"
#     "Texas"
#     "Utah"
#     "Vermont"
#     "Virginia"
#     "Washington"
#     "West_Virginia"
#     "Wisconsin"
#     "Wyoming"
# )

categories=(
    # "Alabama"
    # "Alaska"
    # "Arizona"
    # "Arkansas"
    "California"
    # "Colorado"
    # "Connecticut"
    # "Delaware"
    # "District_of_Columbia"
    "Florida"
    "Georgia"
    # "Hawaii"
    # "Idaho"
    "Illinois"
    # "Indiana"
    # "Iowa"
    # "Kansas"
    # "Kentucky"
    # "Louisiana"
    # "Maine"
    # "Maryland"
    # "Massachusetts"
    "Michigan"
    # "Minnesota"
    # "Mississippi"
    # "Missouri"
    # "Montana"
    # "Nebraska"
    # "Nevada"
    # "New_Hampshire"
    # "New_Jersey"
    # "New_Mexico"
    "New_York"
    # "North_Carolina"
    # "North_Dakota"
    "Ohio"
    # "Oklahoma"
    # "Oregon"
    "Pennsylvania"
    # "Rhode_Island"
    # "South_Carolina"
    # "South_Dakota"
    # "Tennessee"
    "Texas"
    # "Utah"
    # "Vermont"
    # "Virginia"
    # "Washington"
    # "West_Virginia"
    # "Wisconsin"
    # "Wyoming"
)


template="{name}_Fashion:
    path: GoogleLocalData/Fashion/rating_dataset/filtered_{name}.pkl
    text_fields: [text]
    max_behavior_len: 20
    naive_ctr:
        learning_rate: 3e-3
        factor: 0.3
        patience: 0
        train:
            output_dir: GoogleLocalData/Fashion/{name}/ctr
            per_device_train_batch_size: 512
            per_device_eval_batch_size: 512
            logging_strategy: epoch
            save_strategy: epoch
            evaluation_strategy: epoch

"

for name in "${categories[@]}"; do
    content=${template//\{name\}/$name}
    echo "$content" >> "dataset_specific.yaml"
    echo "${name} has been added."
done