import os
import pickle

from dassl.data.datasets import DATASET_REGISTRY, Datum, DatasetBase
from dassl.utils import listdir_nohidden, mkdir_if_missing

from .oxford_pets import OxfordPets


@DATASET_REGISTRY.register()
class Plant_Disease(DatasetBase):

    dataset_dir = "plant_disease/dataset"

    def __init__(self, cfg):
        root = os.path.abspath(os.path.expanduser(cfg.DATASET.ROOT))
        self.dataset_dir = os.path.join(root, self.dataset_dir)
        self.train_dir = os.path.join(self.dataset_dir, "train")
        self.test_dir = os.path.join(self.dataset_dir, "test")
        self.preprocessed = os.path.join(self.dataset_dir, "preprocessed.pkl")
        self.split_fewshot_dir = os.path.join(self.dataset_dir, "split_fewshot")
        mkdir_if_missing(self.split_fewshot_dir)

        if os.path.exists(self.preprocessed):
            with open(self.preprocessed, "rb") as f:
                preprocessed = pickle.load(f)
                train = preprocessed["train"]
                test = preprocessed["test"]
        else:
            train = self.read_data(self.train_dir)
            # Follow standard practice to perform evaluation on the val set
            # Also used as the val set (so evaluate the last-step model)
            test = self.read_data(self.test_dir)

            preprocessed = {"train": train, "test": test}
            with open(self.preprocessed, "wb") as f:
                pickle.dump(preprocessed, f, protocol=pickle.HIGHEST_PROTOCOL)

        num_shots = cfg.DATASET.NUM_SHOTS
        if num_shots >= 1:
            seed = cfg.SEED
            preprocessed = os.path.join(self.split_fewshot_dir, f"shot_{num_shots}-seed_{seed}.pkl")
            
            if os.path.exists(preprocessed):
                print(f"Loading preprocessed few-shot data from {preprocessed}")
                with open(preprocessed, "rb") as file:
                    data = pickle.load(file)
                    train = data["train"]
            else:
                train = self.generate_fewshot_dataset(train, num_shots=num_shots)
                data = {"train": train}
                print(f"Saving preprocessed few-shot data to {preprocessed}")
                with open(preprocessed, "wb") as file:
                    pickle.dump(data, file, protocol=pickle.HIGHEST_PROTOCOL)

        subsample = cfg.DATASET.SUBSAMPLE_CLASSES
        train, test = OxfordPets.subsample_classes(train, test, subsample=subsample)

        super().__init__(train_x=train, val=test, test=test)

    def read_data(self, img_dir):
        image_dir = img_dir
        folders = listdir_nohidden(image_dir, sort=True)
        folders = [f for f in folders]
        items = []

        for label, folder in enumerate(folders):
            imnames = listdir_nohidden(os.path.join(image_dir, folder))
            classname = folder
            for imname in imnames:
                impath = os.path.join(image_dir, folder, imname)
                item = Datum(impath=impath, label=label, classname=classname)
                items.append(item)

        return items
