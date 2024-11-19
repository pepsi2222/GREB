import os
import sys
sys.path.append('..')
sys.path.append('../RecStudio')
from RecStudio.recstudio.utils import parser_yaml
import pickle
from collections import defaultdict

categories = (
    # "Arts_Crafts_and_Sewing",
    # "Automotive",
    # "Baby_Products",
    # "\n",
    # "Cell_Phones_and_Accessories",
    # "\n",
    # "\n",
    # "\n",
    # "\n",
    # "Industrial_and_Scientific",
    # "Musical_Instruments",
    # "Office_Products",
    # "Patio_Lawn_and_Garden",
    # "\n",
    # "\n",
    # "Sports_and_Outdoors",
    # "Tools_and_Home_Improvement",
    # "OnlineRetail",
    # "HM",
    # "Beauty_and_Personal_Care",
    # "\n",
    # "\n",
    # "\n",
    # "Georgia_Fashion",
    # "Illinois_Fashion",
    # "Michigan_Fashion",
    # "New_York_Fashion",
    # "Ohio_Fashion",
    # "Pennsylvania_Fashion",
    # "\n",
    # "\n",
    # "Goodreads",
    # "Books",
    # "\n",
    # "MovieLens",
    # "\n",
    # "\n",
    # "Toys_and_Games",
    # "Video_Games",
    # "\n",
    # "Bili_Cartoon",
    # "Bili_Dance",
    # "Bili_Food",
    # "Bili_Movie",
    # "Bili_Music",
    # "DY",
    # "KU",
    # "QB",
    # "TN",
    # "Yelp",
    # "\n",
    # "Arkansas_Food",
    # "\n",
    # "Delaware_Food",
    # "District_of_Columbia_Food",
    # "Hawaii_Food",
    # "Idaho_Food",
    # "Maine_Food",
    # "\n",
)

categories = (
    "CDs_and_Vinyl",
    "Electronics",
    "Grocery_and_Gourmet_Food",
    "Health_and_Household",
    "Home_and_Kitchen",
    "Pet_Supplies",
    "Software",
    "Clothing_Shoes_and_Jewelry",
    "California_Fashion",
    "Florida_Fashion",
    "Texas_Fashion",
    "Kindle_Store",
    "Movies_and_TV",
    "Steam",
    "Bili_2M",
    "Alabama_Food",
    "Connecticut_Food",
    "Mississippi_Food",
)


dataset_specific = parser_yaml('dataset_specific.yaml')

stat = defaultdict(dict)

for c in categories:
    if c != '\n':
        # path = os.path.join(os.getenv('DATA_MOUNT_DIR'), dataset_specific[c]['path'])
        path = os.path.join(os.getenv('DATA_MOUNT_DIR'), dataset_specific[c]['pure_dataset'])
        with open(path, 'rb') as f:
            # trn_data = pickle.load(f)
            # val_data = pickle.load(f)
            dataset = pickle.load(f)
            stat[c] = {
                'user': dataset.num_users,
                'item': dataset.num_items,
                'inter': dataset.num_inters,
                'feat': len(dataset.user_feat.columns) + len(dataset.item_feat.columns)
            }
            print(c, stat[c], dataset.user_feat.columns, dataset.item_feat.columns)
    else:
        pass

with open(f'stats_source.txt', 'w') as f:
    for m in ['user', 'item', 'inter', 'feat']:
        f.write(f'{m}: \n')
        for c in categories:
            if c != '\n':
                try:
                    f.write(f'{stat[c][m]}\n')
                except Exception as e:
                    print(c, e)
                    f.write('\n')
            else:
                f.write('\n')