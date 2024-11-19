#!/bin/bash
# some categories becoming empty after 5-core processing.

# Category list
categories=(
    "All_Beauty"
    "Amazon_Fashion"
    "Appliances"
    "Arts_Crafts_and_Sewing"
    "Automotive"
    "Baby_Products"
    "Beauty_and_Personal_Care"
    "Books"
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

# Base URL
base_url="https://datarepo.eng.ucsd.edu/mcauley_group/data/amazon_2023/benchmark/5core/rating_only/"

# Download each category
for category in "${categories[@]}"
do
    url="${base_url}${category}.csv.gz"
    wget -O "${category}.csv.gz" "${url}"
    gzip -d "${category}.csv.gz"
done
rm *.gz
