#!/bin/bash

template="url: /data1/home/xingmei/GRE/data/GoogleLocalData/Food
user_id_field: &u user_id:token # TODO: comments for &u and *u
item_id_field: &i gmap_id:token
rating_field: &r rating:float
time_field: &t timestamp:float
time_format: ~

encoding_method: utf-8

inter_feat_name: review/{name}.csv
inter_feat_field: [*u, *i, *r, *t]
inter_feat_header: 0

user_feat_name: ~
user_feat_field: ~
user_feat_header: ~

item_feat_name: [meta/{name}.csv]
item_feat_field: [[*i, name:text, category:token_seq:\"|\"]]
item_feat_header: 0

field_separator: \"\t\"
min_user_inter: 5
min_item_inter: 0
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
binarized_rating_thres: 3.0

drop_dup: True
max_seq_len: 20

float_field_preprocess: ~ #[price:KBinsDiscretizer(encode='ordinal')]

save_cache: False # whether to save processed dataset to cache.
"

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
    "Georgia"
    "Hawaii"
    "Idaho"
    "Illinois"
    "Indiana"
    "Iowa"
    "Kansas"
    "Kentucky"
    "Louisiana"
    "Maine"
    "Maryland"
    "Massachusetts"
    "Michigan"
    "Minnesota"
    "Mississippi"
    "Missouri"
    "Montana"
    "Nebraska"
    "Nevada"
    "New_Hampshire"
    "New_Jersey"
    "New_Mexico"
    "New_York"
    "North_Carolina"
    "North_Dakota"
    "Ohio"
    "Oklahoma"
    "Oregon"
    "Pennsylvania"
    "Rhode_Island"
    "South_Carolina"
    "South_Dakota"
    "Tennessee"
    "Texas"
    "Utah"
    "Vermont"
    "Virginia"
    "Washington"
    "West_Virginia"
    "Wisconsin"
    "Wyoming"
)


for name in "${categories[@]}"; do
    content=${template//\{name\}/$name}
    echo "$content" > "${name}.yaml"
    echo "${name}.yaml has been created."
done
