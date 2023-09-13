import os
import pickle
import pandas as pd
import numpy as np

from dassl.data.datasets import DATASET_REGISTRY, Datum, DatasetBase
from dassl.utils import mkdir_if_missing

from .oxford_pets import OxfordPets


@DATASET_REGISTRY.register()
class ISIC(DatasetBase):

    dataset_dir = "isic"

    def __init__(self, cfg):
        root = os.path.abspath(os.path.expanduser(cfg.DATASET.ROOT))
        self.dataset_dir = os.path.join(root, self.dataset_dir)
        self.train_image_dir = os.path.join(self.dataset_dir, "train/ISIC2018_Task3_Training_Input")
        self.val_image_dir = os.path.join(self.dataset_dir, "val/ISIC2018_Task3_Validation_Input")
        self.test_image_dir = os.path.join(self.dataset_dir, "test/ISIC2018_Task3_Test_Input")
        self.train_csv_path = os.path.join(self.dataset_dir, "train/ISIC2018_Task3_Training_GroundTruth/ISIC2018_Task3_Training_GroundTruth.csv")
        self.val_csv_path = os.path.join(self.dataset_dir, "val/ISIC2018_Task3_Validation_GroundTruth/ISIC2018_Task3_Validation_GroundTruth.csv")
        self.test_csv_path = os.path.join(self.dataset_dir, "test/ISIC2018_Task3_Test_GroundTruth/ISIC2018_Task3_Test_GroundTruth.csv")
        self.split_fewshot_dir = os.path.join(self.dataset_dir, "split_fewshot")
        mkdir_if_missing(self.split_fewshot_dir)

        train = self.read_data(self.train_image_dir, self.train_csv_path)
        val = self.read_data(self.val_image_dir, self.val_csv_path)
        test = self.read_data(self.test_image_dir, self.test_csv_path)

        num_shots = cfg.DATASET.NUM_SHOTS
        if num_shots >= 1:
            seed = cfg.SEED
            preprocessed = os.path.join(self.split_fewshot_dir, f"shot_{num_shots}-seed_{seed}.pkl")
            
            if os.path.exists(preprocessed):
                print(f"Loading preprocessed few-shot data from {preprocessed}")
                with open(preprocessed, "rb") as file:
                    data = pickle.load(file)
                    train, val = data["train"], data["val"]
            else:
                train = self.generate_fewshot_dataset(train, num_shots=num_shots)
                val = self.generate_fewshot_dataset(val, num_shots=min(num_shots, 4))
                data = {"train": train, "val": val}
                print(f"Saving preprocessed few-shot data to {preprocessed}")
                with open(preprocessed, "wb") as file:
                    pickle.dump(data, file, protocol=pickle.HIGHEST_PROTOCOL)

        subsample = cfg.DATASET.SUBSAMPLE_CLASSES
        train, val, test = OxfordPets.subsample_classes(train, val, test, subsample=subsample)

        super().__init__(train_x=train, val=val, test=test)

    
    def read_data(self, img_path, csv_path):
        data_info = pd.read_csv(csv_path, skiprows=[0], header=None)
        columns = list(pd.read_csv(csv_path).columns)[1:]
        image_name = np.asarray(data_info.iloc[:, 0])
        labels = np.asarray(data_info.iloc[:, 1:])
        labels = (labels!=0).argmax(axis=1)
        items = []

        for i in range(len(image_name)):
          impath = os.path.join(img_path,image_name[i]+'.jpg')
          label = labels[i]
          classname = columns[label]
          item = Datum(impath=impath, label=label, classname=classname)
          items.append(item)

        return items
