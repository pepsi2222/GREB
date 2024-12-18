#!/bin/bash

template="url: /data1/home/xingmei/GRE/data/AmazonReviews2023/
user_id_field: &u user_id:token # TODO: comments for &u and *u
item_id_field: &i parent_asin:token
rating_field: &r rating:float
time_field: &t timestamp:float
time_format: ~

encoding_method: utf-8

inter_feat_name: pureID_inter/new_{name}.csv
inter_feat_field: [*u, *i, *r, *t]
inter_feat_header: 0

user_feat_name: ~
user_feat_field: ~
user_feat_header: ~

item_feat_name: [meta_categories/{name}.csv]
item_feat_field: [[*i, categories:token_seq:\"|||\", title:text, description:text, features:text]]
item_feat_header: 0

field_separator: \"\t\"
min_user_inter: 5
min_item_inter: 5
field_max_len: ~      # a YAML-format dict, for example
# field_max_len:
#   age: 1
#   gender: 1
#   occupation: 1
low_rating_thres: ~   # low rating threshold, which is used for drop low rating interactions
# drop_low_rating: True # if true, the interactions with rating lower than \`rating_thres\` would be dropped.

# negative rating threshold, interactions with rating below than the threshold would be regarded as negative interactions.
# Note that when \`drop_low_rating\` is True, only interactions with rating above \`low_rating_thres\` and below \`negative_rating_thres\`
# would be regared as negative interactions.
# The threshold value should be larger than \`low_rating_thres\`. If not, the threshold would be invalid, which means all interactions kept
# would be regarded as positives.
# negative_rating_thres: 0.0

# \`binarized_rating\` controls whether to binarize the rating to 0/1 with the \`rating_thres\`.
# If true, ratings above \`rating_thres\` would be mapped as 1 and ratings above \`rating_thres\` would be mapped as 0;
# If false, the ratings would not be changed
binarized_rating_thres: 0.0

drop_dup: True
max_seq_len: 20

float_field_preprocess: ~ #[price:KBinsDiscretizer(encode='ordinal')]

save_cache: False # whether to save processed dataset to cache.
"

categories=(
    # "All_Beauty"
    # "Amazon_Fashion"
    # "Appliances"
    # "Arts_Crafts_and_Sewing"
    # "Automotive"
    # "Baby_Products"
    # "Beauty_and_Personal_Care"
    # "Books"
    "CDs_and_Vinyl"
    "Cell_Phones_and_Accessories"
    "Clothing_Shoes_and_Jewelry"
    "Digital_Music"
    "Electronics"
    "Gift_Cards"
    "Grocery_and_Gourmet_Food"
    "Handmade_Products"
    "Health_and_Household"
    "Health_and_Personal_Care"
    "Home_and_Kitchen"
    "Industrial_and_Scientific"
    "Kindle_Store"
    "Magazine_Subscriptions"
    "Movies_and_TV"
    "Musical_Instruments"
    "Office_Products"
    "Patio_Lawn_and_Garden"
    "Pet_Supplies"
    "Software"
    "Sports_and_Outdoors"
    "Subscription_Boxes"
    "Tools_and_Home_Improvement"
    "Toys_and_Games"
    "Video_Games"
)


for name in "${categories[@]}"; do
    content=${template//\{name\}/$name}
    echo "$content" > "${name}.yaml"
    echo "${name}.yaml has been created."
done
