#!/bin/bash

# Category list
categories=(
    # "All_Beauty"
    # "Amazon_Fashion"
    # "Appliances"
    "Arts_Crafts_and_Sewing"
    "Automotive"
    "Baby_Products"
    "Beauty_and_Personal_Care"
    "Books"
    # "CDs_and_Vinyl"
    "Cell_Phones_and_Accessories"
    # "Clothing_Shoes_and_Jewelry"
    # "Digital_Music"
    # "Electronics"
    # "Gift_Cards"
    # "Grocery_and_Gourmet_Food"
    # "Handmade_Products"
    # "Health_and_Household"
    # "Health_and_Personal_Care"
    # "Home_and_Kitchen"
    "Industrial_and_Scientific"
    # "Kindle_Store"
    # "Magazine_Subscriptions"
    # "Movies_and_TV"
    "Musical_Instruments"
    "Office_Products"
    "Patio_Lawn_and_Garden"
    # "Pet_Supplies"
    # "Software"
    "Sports_and_Outdoors"
    # "Subscription_Boxes"
    "Tools_and_Home_Improvement"
    "Toys_and_Games"
    "Video_Games"
)


for category in "${categories[@]}"
do
    python "$1.py" $category
    echo "${category} has been processed."
done
