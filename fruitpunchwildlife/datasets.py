from sklearn.model_selection import train_test_split
import pickle
import os
import pandas as pd
import numpy as np
import json
import yaml
from pathlib import Path
import shutil


from pprint import pprint
import copy


class DatasetLoader:
    def __init__(self, dataset_path: str, dataset_config: dict):
        self.dataset_path = dataset_path
        self.config = copy.deepcopy(dataset_config)
        self.df_dict = {}
        self.df = None
        self.new_index = None

    def prepare_datasets(self, drop_na=True):
        self._load_dataset(drop_na=drop_na)

        print("Prepare media path")
        self._prepare_media_path()

        print("Unify dataset")
        self.df = pd.concat([self.df_dict[dataset]
                            for dataset in self.df_dict])

        print("Prepare deploymentID_category")
        self.prepare_trapper_regions()

    def _load_dataset(self, drop_na):
        for dataset in self.config:
            print(dataset)

            df = pd.read_csv(self.config[dataset]["metadata"])
            total = len(df)

            if drop_na:
                print(f"Drop na: {len(df[df.isnull().any(axis=1)])}/{total}")
                df = df.dropna()

                self.config[dataset]['drop_na'] = len(
                    df[df.isnull().any(axis=1)])
            else:
                self.config[dataset]['drop_na'] = None

            print("Store temporary dataset")
            self.df_dict[dataset] = df
            print()

    def generate_index(self):
        list_names = self._collect_categories(self.df_dict)
        species_list = self._unify_categories(list_names)
        self._create_index(species_list)

    def _collect_categories(self, df_dict: dict):

        list_names = []

        for dataset in df_dict:
            df = df_dict[dataset]
            if df is None:
                print(f"Skip {dataset}")
                continue

            for name in df[['commonName', 'class_id']].value_counts().keys():
                list_names.append(
                    [name[0], name[1], df[['commonName', 'class_id']].value_counts()[name]])

        list_names.sort()

#         print ('commonName, class_id, counts')
#         pprint(list_names)
        return list_names

    def _unify_categories(self, list_names: list):
        species_dict = {name: 0 for name, _, _ in list_names}
        total = 0
        for name, class_id, amount in list_names:
            species_dict[name] += amount
            total += amount

        species_list = list(species_dict.items())
        species_list.sort(key=lambda x: x[1], reverse=True)
#         pprint(species_list[:10])
        return species_list

    def _create_index(self, species_list: tuple):
        species_class = []
        species_id = 0
        for name, amount in species_list:
            species_class.append([name, species_id, amount])
            species_id += 1

        self.new_index = pd.DataFrame(np.array(species_class),
                                      columns=['commonName', 'class_id', 'count'])

    def save_index(self, path_to_csv):
        self.new_index.to_csv(path_to_csv, index=False)

    def load_index(self, path_to_csv):
        self.new_index = pd.read_csv(path_to_csv)

    def _prepare_media_path(self):
        for dataset in self.df_dict:
            #             print(dataset)
            df = self.df_dict[dataset]
            path = self.config[dataset]['path']

            # add path to the images relative to the DATASET path
            df['media_path'] = df.apply(lambda row: f"{path}/{row['mediaID']}.jpg", axis=1)
            df['annotation_path'] = df.apply(lambda row: f"{path}/{row['mediaID']}.txt", axis=1)
            df['dataset'] = dataset

    def apply_new_index(self):
        if self.new_index is None:
            print("Please generate or load a index to apply")
            return

        new_index_dict = {row[0]: row[1]
                          for index, row in self.new_index.iterrows()}

        print("Creating new column: class_id_updated")
        self.df['class_id_updated'] = self.df.apply(
            lambda row: int(new_index_dict[row["commonName"]]), axis=1)

        print("Updating references in bbox.")
        print("Creating new column: yolo_bboxes_updated")
        self.df['yolo_bboxes_updated'] = self.df.apply(
            lambda row: self.update_yolo_bboxes(row), axis=1)

    def update_yolo_bboxes(self, row):
        yolo_detection = eval(row['yolo_bboxes'])
        yolo_detection_updated = []
        for detection in yolo_detection:
            detection["class_id"] = row['class_id_updated']
            yolo_detection_updated.append(detection)
    #     print(yolo_detection)
        return json.dumps(yolo_detection_updated)

    def prepare_trapper_regions(self):
        self.df['deploymentID_region'] = self.df.apply(
            lambda row: row['deploymentID'].replace('_', '-'), axis=1)
        self.df['deploymentID_region'] = self.df.apply(
            lambda row: row['deploymentID_region'].split('-')[0], axis=1)
        print('Created deploymentID_region')

        self.df['deploymentID_category'] = self.df['deploymentID_region'].astype(
            'category')
        print('Created deploymentID_category')

    def create_balanced_dataset(self, main_fields=['commonName', 'deploymentID_category'], class_id_filter=None, max_samples=100):
        print("Balanced using the following fields:", main_fields)
#         if 'deploymentID_category' in main_fields:
#             self.prepare_trapper_regions()

        if class_id_filter is not None:
            print("Filter dataset by :", class_id_filter)
            df_filtered = self._filter_dataset(class_id_filter)
        else:
            print("Filter dataset by : Skip")
            df_filtered = self.df

        df_balanced = self._balance_dataset(
            df_filtered, main_fields, max_samples)

        return df_balanced

    def _filter_dataset(self, class_id_filter):
        pprint(self.df[['commonName', 'class_id_updated',]].value_counts())
        df_filtered = self.df[self.df['class_id_updated'] <= class_id_filter]
        pprint(df_filtered[['commonName', 'class_id_updated',]].value_counts())

        return df_filtered

    def _balance_dataset(self, df, main_fields, max_samples):

        # Calculate the count of samples in each category combination.
        counts = df.groupby(main_fields).size().reset_index(name='count')
#         pprint(counts)

        # Find the minimum count among all category combinations.
        min_count = counts['count'].min()
#         pprint(min_count)

        # Initialize an empty dataframe to store the balanced dataset.
        balanced_df = pd.DataFrame(columns=df.columns)

        # Iterate through each category combination and sample the required number of rows to achieve balance.
        for _, row in counts.iterrows():
            common_name = row['commonName']
            deployment_category = row['deploymentID_category']
            count = row['count']

            if count == 0:
                continue

            subset = df[(df['commonName'] == common_name) & (
                df['deploymentID_category'] == deployment_category)]
            if len(subset) < max_samples:
                balanced_subset = subset
            else:
                balanced_subset = subset.sample(
                    n=max_samples, replace=True, random_state=42)

            balanced_df = pd.concat([balanced_df, balanced_subset])
        df = balanced_df
        return df

    # After executing these steps, the balanced_df dataframe will contain a balanced dataset based on the "commonName" and "deploymentID_category" columns. Each category combination will have the same number of samples, equal to the count of the minority category combination.

    def create_annotations(self):
        print("Generating annotation files")
#         print (max([len(json.loads(row['yolo_bboxes_updated'])) for index, row in df.iterrows()]))
        for index, row in self.df.iterrows():
            with open(row['annotation_path'], 'w') as f:
                #         print(row['annotation_path'])
                yolo_detection = json.loads(row['yolo_bboxes_updated'])
                for detection in yolo_detection:

                    annotation = f"{detection['class_id']} {detection['x']} {detection['y']} {detection['w']} {detection['h']}\n"
                    f.write(annotation)
        #             print(annotation)


def split_balanced_dataset(df, stratify_fields=None, pickle_filename=None):
    # stratify_fields=['commonName','deploymentID_category']
    # pickle_filename = 'split_sample.pickle'

    stratify = None
    if stratify_fields is not None:
        stratify = df[stratify_fields]

    # Split the dataset into train-valid-test splits
    train_images, val_images, train_annotations, val_annotations = train_test_split(
        df['media_path'].to_list(), df['annotation_path'].to_list(), test_size=0.2, random_state=42, stratify=stratify)
    val_images, test_images, val_annotations, test_annotations = train_test_split(
        val_images, val_annotations, test_size=0.5, random_state=42)

    split_dict = {'images': {'test': test_images, 'val': val_images, 'train': train_images},
                  'labels': {'test': test_annotations, 'val': val_annotations, 'train': train_annotations}}

    if pickle_filename:
        #         print ("Store in ", pickle_filename)
        with open(pickle_filename, 'wb') as f:
            pickle.dump(split_dict, f)

    return split_dict


def prepare_yolov58(split_dict, df):

    list_files = [('train.txt', split_dict['images']['train']), 
                  ('val.txt', split_dict['images']['val']), 
                  ('test.txt', split_dict['images']['test']),]

    # Store temporary files
    for filename, images_set in list_files:
        with open(f"{filename}", 'w') as f:
            for images in images_set:
                f.write(f"{images}\n")

    # Create classes dict
    classes = {}
    for index, row in df.iterrows():
        if not row['class_id_updated'] in classes:
            classes[row['class_id_updated']] = row['commonName']

    data_yaml = {
        "names": classes,
        "nc": len(classes),
        # this points to the base folder you find train.txt and the others...
        "path": str(Path("./").resolve()),
        "train": "train.txt",
        "val": "val.txt",
        "test": "test.txt"
    }

    data_yaml_path = Path(
        f"{str(Path('./').resolve())}/yolo_info.yaml").resolve()
    with open(data_yaml_path, 'w') as f:
        yaml.dump(data_yaml, f, sort_keys=False, default_flow_style=False)

    return data_yaml_path


def prepare_yolonas(split_dict, target_folder):
    dict_folders = {}
    for category in ['images', 'labels']:
        dict_folders[category] = {}
        for stage in ['test', 'train', 'val']:
            dict_folders[category][stage] = os.path.join(
                target_folder, category, stage)
            os.makedirs(dict_folders[category][stage], exist_ok=True)
            print(dict_folders[category][stage])
            for filename in split_dict[category][stage]:
                shutil.copy(filename, os.path.join(
                    dict_folders[category][stage], filename.split('/')[-1]))
