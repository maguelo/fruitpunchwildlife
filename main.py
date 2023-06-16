from fruitpunchwildlife import datasets
from pprint import pprint 


# Install the library executing in collab
# !pip install git+https://github.com/maguelo/fruitpunchwildlife



if __name__=='__main__':
    DATASET_PATH = "/home/maguelo/Workspace/fruitpunch/megadetector/dataset" #"dataset"  # Path to the dataset

    DATASET_DICT={
                    "trapper_photos_13":{"metadata": f"{DATASET_PATH}/trapper_photos_13/metadata.csv", "path": f"{DATASET_PATH}/trapper_photos_13"},
                    "trapper_photos_2":{"metadata": f"{DATASET_PATH}/trapper_photos_2/metadata.csv", "path": f"{DATASET_PATH}/trapper_photos_2"},
                    "trapper_photos_6":{"metadata": f"{DATASET_PATH}/trapper_photos_6/metadata.csv", "path": f"{DATASET_PATH}/trapper_photos_6"},
                }
    
    data_loader = datasets.DatasetLoader(DATASET_PATH, DATASET_DICT)
    data_loader.prepare_datasets()
    data_loader.generate_index()
    data_loader.apply_new_index()
    data_loader.create_annotations()

    MAIN_FIELDS = ['commonName','deploymentID_category']

    balanced_df=data_loader.create_balanced_dataset(main_fields=MAIN_FIELDS,class_id_filter=10, max_samples=10)



    list_region_name = [(val[0],val[1], val[2], cnt) for val, cnt in balanced_df[MAIN_FIELDS+['class_id_updated']].value_counts().items()]
                # list_values = {val for val, cnt in df_full.deploymentID_region.value_counts().items()}
    list_region_name=sorted(list_region_name, key=lambda x: x[0])
    pprint(list_region_name)


    split_dict=datasets.split_balanced_dataset(balanced_df)        

    # Generate dataset.yaml
    print(datasets.prepare_yolov58(split_dict,balanced_df))

    #Work in progress already generated the folders with the images and labels with the new category id
    datasets.prepare_yolonas(split_dict, 'test_dataset')


