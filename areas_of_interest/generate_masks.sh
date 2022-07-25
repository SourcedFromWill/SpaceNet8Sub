#!/bin/bash
# RUN FROM THIS DIRECTORY

cd ..

python data_prep/geojson_prep.py --root_dir areas_of_interest --aoi_dirs Germany_Training_Public Louisiana-East_Training_Public

python data_prep/create_masks.py --root_dir areas_of_interest --aoi_dirs Germany_Training_Public Louisiana-East_Training_Public

# NO GENERATION OF NEW CSVS