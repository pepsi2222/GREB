#!/bin/bash

template="url: /data1/home/xingmei/GRE/data/Bili/
user_id_field: &u uid:token # TODO: comments for &u and *u
item_id_field: &i iid:token
rating_field: ~
time_field: &t timestamp:float
time_format: ~

encoding_method: utf-8

inter_feat_name: Downstream_datasets/{name}/{name}_pair.csv
inter_feat_field: [*i, *u, *t]
inter_feat_header: ~

user_feat_name: ~
user_feat_field: ~
user_feat_header: ~


item_feat_name: [Downstream_datasets/{name}/new_item.csv]
item_feat_field: [[*i, en_title:text]]
item_feat_header: 0


field_separator: \",\"
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
binarized_rating_thres: 0.0

drop_dup: True
max_seq_len: 20


float_field_preprocess: ~ #[price:KBinsDiscretizer(encode='ordinal')]

save_cache: False # whether to save processed dataset to cache.
"

categories=(
    "Bili_Cartoon"
    "Bili_Dance"
    "Bili_Food"
    "Bili_Movie"
    "Bili_Music"
    "DY"
    "KU"
    "QB"
    "TN"
)


for name in "${categories[@]}"; do
    content=${template//\{name\}/$name}
    echo "$content" > "${name}.yaml"
    echo "${name}.yaml has been created."
done
