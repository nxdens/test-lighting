import logging
from optparse import Option
import sys
import os
from treelib import Node, Tree
from pyrsistent import optional
import torch
import torch.utils.data
from torch.utils.data import Dataset, DataLoader, IterableDataset
import torchio as tio
import monai.transforms
import radio.data as radata
from radio.settings.pathutils import is_dir_or_symlink, PathType
from radio.data.datatypes import SpatialShapeType
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union, cast
from ..misc import nifti_helpers
import pandas as pd
import numpy as np
import re
import collections


Sample = List[Tuple[Path, Any]]
OneSample = Union[Dict[str, Tuple[Any, ...]], Tuple[Any, ...]]

# TODO: make something to compile subject ID, scan ID and target


class DepDataModule(radata.BaseDataModule):
    """
    This class will be based on the folder dataset
    in radio for our specific project
    DO NOT use this with a queue yet
    """

    # CHANGE THIS FOR DIFFERENT ID PATTERNS. USED IN prepare_data()
    ID_PATTERN = "ABD-[A-Z]+-\d\d\d\d"

    def __init__(
        self,
        *args,
        root: PathType,
        base_csv: pd.DataFrame,
        study: str = "",
        subj_dir: str = "Public/data",
        data_dir: str = "",
        subject_dicts: Optional[
            Union[dict[str, list[str]], dict[str, pd.DataFrame]]
        ] = None,
        modalities: Optional[List[str]] = None,
        labels: Optional[List[str]] = None,
        train_transforms: Optional[tio.Transform] = None,
        val_transforms: Optional[tio.Transform] = None,
        test_transforms: Optional[tio.Transform] = None,
        use_augmentation: bool = True,
        use_preprocessing: bool = True,
        resample: str = None,
        patch_size: Optional[SpatialShapeType] = None,
        probability_map: Optional[str] = None,
        create_custom_probability_map: bool = False,
        label_name: Optional[str] = None,
        label_probabilities: Optional[Dict[int, float]] = None,
        queue_max_length: int = 256,
        samples_per_volume: int = 16,
        batch_size: int = 32,
        shuffle_subjects: bool = True,
        shuffle_patches: bool = True,
        num_workers: int = 0,
        pin_memory: bool = True,
        start_background: bool = True,
        drop_last: bool = False,
        num_folds: int = 2,
        val_split: Union[int, float] = 0.2,
        dims: Tuple[int, int, int] = (256, 256, 256),
        seed: int = 41,
        verbose: bool = False,
        transform: Optional[Callable] = None,
        **kwargs,
    ) -> None:
        super().__init__(
            *args,
            root=root,
            train_transforms=train_transforms,
            val_transforms=val_transforms,
            test_transforms=test_transforms,
            batch_size=batch_size,
            shuffle=shuffle_subjects,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=drop_last,
            num_folds=num_folds,
            val_split=val_split,
            seed=seed,
            **kwargs,
        )
        # input dict should have the same keys
        self.data = pd.read_csv(base_csv)

        self.subject_list = None
        if subject_dicts:
            self.subject_list = {"train": None, "test": None, "val": None}
            if isinstance(subject_dicts["train"], list):
                # Subject_dataframe should contain an column indicating which fold
                for key in self.subject_list.key():
                    self.subject_list[key] = pd.DataFrame(
                        np.array(subject_dicts[key]), columns=["subjectFolder"]
                    )
            else:  # dict of pd.DataFrame's
                for key in self.subject_list.key():
                    self.subject_list[key] = subject_dicts[key]

    def prepare_data(self, *args: Any, **kwargs: Any) -> None:
        """
        Verify data directory exists.
        Verify if test/train/val splitted.
        """
        if not is_dir_or_symlink(self.root):
            raise OSError("Study data directory not found!")
        self.parse_root(self.root)
        self.match_ID_col()
        # self.depth = self.find_tree_depth()
        # print("depth", self.depth)
        self.find_images()
        self.check_splits()
        # make dataframe somehow

    def setup(self, *args: Any, stage: Optional[str] = None, **kwargs: Any) -> None:
        if stage == None or stage == "fit":
            train_transforms = (
                self.default_transforms(stage="fit")
                if self.train_transforms is None
                else self.train_transforms
            )
        if stage == None or stage == "val":
            val_transforms = (
                self.default_transforms(stage="fit")
                if self.val_transforms is None
                else self.val_transforms
            )
        if stage == None or stage == "fit":
            train_list = self.get_subjects(fold="train")
            self.train_dataset = tio.SubjectsDataset(
                train_list, transform=train_transforms
            )
            self.train_loader = self.dataloader(self.train_dataset)

        if stage == None or stage == "fit":
            val_list = self.get_subjects(fold="val")
            self.val_dataset = tio.SubjectsDataset(val_list, transform=val_transforms)
            self.val_loader = self.dataloader(self.val_dataset)

        if stage == None or stage == "fit":
            test_list = self.get_subjects(fold="test")
            self.test_dataset = tio.SubjectsDataset(test_list)
            self.test_loader = self.dataloader(self.test_dataset)

    def match_ID_col(self):
        # find column names with 'id' followed by not a letter
        id_cols = [
            col
            for col in self.data.columns.values
            if re.search("id^[a-z]|id$", col.lower())
        ]
        self.id_col = []  # length 1 string array
        for col in id_cols:
            if re.search(self.ID_PATTERN, str(self.data[col].values[2])):
                self.id_col = str(col)
                break
        # none found need to manually find data

    def find_tree_depth(self) -> int:
        # returns true if id pattern matches a file(?)
        depth = 1
        start = self.file_tree.get_node(str(self.root))
        depth = self._breath_first_id_search(self.file_tree, start, self.ID_PATTERN)
        return depth

    @staticmethod
    def _breath_first_id_search(tree, root, pattern) -> int:
        q = collections.deque()
        q.append(root)
        depth = 1

        def _check_node(node, pattern) -> bool:
            return str(node.data.file_typ) == "image" and re.search(pattern, node.tag)

        while len(q) > 0:
            node = q.popleft()
            if _check_node(node, pattern):
                split = node.identifier.split("/")
                for i, dirname in enumerate(split):
                    if dirname == root.tag:
                        return len(split) - i
            for child in node.fpointer:
                q.append(tree.get_node(str(child)))
        return -1

    def find_images(self):
        self.subjects_files_dict = {}
        for item in self.data[self.id_col]:
            self.subjects_files_dict[str(item)] = []
        for node in self.file_tree.all_nodes_itr():
            match = re.search(self.ID_PATTERN, node.tag)
            if match and node.data.file_typ == "image":
                try:
                    self.subjects_files_dict[match.group()].append(node)
                except:
                    pass

    def check_splits(self, subsets: Tuple[float, float, float] = (0.8, 0.1, 0.1)):

        if not self.subject_list:
            self.subject_list = {"train": None, "test": None, "val": None}
            ids = np.array(list(set(self.data[str(self.id_col)].values)))
            np.random.shuffle(ids)
            first_split = int(ids.shape[0] * subsets[0])
            second_split = int(ids.shape[0] * (subsets[0] + subsets[1]))

            ids_dict = {
                "train": ids[0:first_split],
                "val": ids[first_split:second_split],
                "test": ids[second_split:],
            }
            for key in ["train", "test", "val"]:
                self.subject_list[key] = pd.DataFrame(
                    ids_dict[key], columns=[self.id_col]
                )

    def teardown(self):
        pass

    def train_dataloader(self):
        return self.train_loader

    def val_dataloader(self):
        return self.val_loader

    def test_dataloader(self):
        return self.test_loader

    @staticmethod
    def tio_subjects_to_dataframe(
        subjects: Union[list[tio.Subject], tio.SubjectsDataset]
    ) -> pd.DataFrame:
        if isinstance(subjects, list):
            dataset = tio.SubjectsDataset(subjects)
        else:
            dataset = subjects

        data = []
        for sub in dataset.dry_iter():
            data.append(pd.DataFrame(sub))

        return pd.concat(data)

    @staticmethod
    def dataframe_to_tio_subjects(data: pd.DataFrame) -> List[tio.Subject]:
        subject_list = []
        for item in data.iterrows():
            subject_list.append(tio.Subject(item[1].to_dict()))

        return subject_list

    @staticmethod
    def get_IDs(root, extra_folder: str = ""):
        folders = os.listdir(root)
        sub_names = []
        scan_names = []
        temp = []

        for root, dirs, files in os.walk(root):
            for name in dirs:
                base = os.path.basename(root)
                if base in sub_names:
                    if base in scan_names:
                        pass  # TODO: figure out what to do with the extra folders
                    else:
                        scan_names.append(name)
                        temp.append([base, name])
                else:
                    sub_names.append(name)

        combined_sub_scan = np.array(temp)
        return pd.DataFrame(
            {
                "subjectFolder": combined_sub_scan[:, 0],
                "scanFolder": combined_sub_scan[:, 1],
            }
        )

    def get_targets(self, target_col):
        if self.subjectCol is None or self.scanCol is None:
            self.subjectCol = self.data.columns[
                np.where(self.data.isin([self.IDs.iloc[0]["subjectFolder"]]))[1][0]
            ]
            self.scanCol = self.data.columns[
                np.where(self.data.isin([self.IDs.iloc[0]["scanFolder"]]))[1][0]
            ]
        target_vals = []
        for index, row in self.IDs.iterrows():
            target_vals.append(
                self.data[
                    self.data[self.subjectCol] == row["subjectFolder"]
                    and self.data[self.scanCol] == row["scanFolder"]
                ]
            )
        self.subset_data = pd.DataFrame(
            {
                "subjectFolder": self.IDs["subjectFolder"],
                "scanFolder": self.IDs["scanFolder"],
                "label": target_vals,
            }
        )

    def parse_root(self, root_dir):
        """parses the folder structure of the root path given. designed for ABD raw folder.

        Args:
            root (str): path of the starting folder for the walk

        Returns:
            Tree: tree of files and folders in the root folder
        """
        self.file_tree = Tree()
        ignore_files = [".DS_Store"]
        self.file_tree.create_node(
            os.path.basename(root_dir),
            str(root_dir),
            data=file_type("directory", root_dir),
        )
        for root, dirs, files in os.walk(root_dir):
            for name in dirs:
                self.file_tree.create_node(
                    name,
                    os.path.join(root, name),
                    parent=root,
                    data=file_type("directory", os.path.join(root, name)),
                )
            for name in files:
                if not name in ignore_files:
                    self.file_tree.create_node(
                        name,
                        os.path.join(root, name),
                        parent=root,
                        data=file_type("file", os.path.join(root, name)),
                    )
        return self.file_tree

    def get_subjects(self, fold: str = "train") -> List[tio.Subject]:
        train_subjs, test_subjs, val_subjs = self.get_subjects_lists()
        if fold == "train":
            subs = train_subjs
        elif fold == "test":
            subs = test_subjs
        else:
            subs = val_subjs

        return subs

    def get_subjects_lists(self) -> List[tio.Subject]:
        def _get_subjects_list(
            data: pd.DataFrame, subject_list: pd.DataFrame
        ) -> Tuple[List[tio.Subject], List[tio.Subject], List[tio.Subject]]:
            # Need to drop na values somewhere before this
            if subject_list is not None:
                subset = data[data[self.id_col].isin(subject_list[self.id_col])]
            else:
                subset = data
            dummy_image = tio.ScalarImage(tensor=torch.rand(1, 128, 128, 128))
            subjects = []
            for sub in subset.iterrows():
                sub_dict = {"dummy": dummy_image}
                image_path = self.subjects_files_dict[sub[1][self.id_col]][0].data.path
                modality = re.search(".*(?=\.)", str(image_path)).group().split("_")[-1]
                sub_dict[modality] = tio.ScalarImage(image_path)
                sub_dict = dict(sub_dict, **(sub[1].to_dict()))
                subjects.append(tio.Subject(sub_dict))
            return subjects

        train_list = _get_subjects_list(
            data=self.data, subject_list=self.subject_list["train"]
        )
        val_list = _get_subjects_list(
            data=self.data, subject_list=self.subject_list["val"]
        )
        test_list = _get_subjects_list(
            data=self.data, subject_list=self.subject_list["test"]
        )
        return train_list, val_list, test_list

    def dataloader(
        self,
        dataset: torch.utils.data.dataloader,
        batch_size: Optional[int] = None,
        shuffle: Optional[bool] = None,
        num_workers: Optional[int] = None,
        pin_memory: Optional[bool] = None,
        drop_last: Optional[bool] = None,
    ) -> DataLoader:
        """
        Instantiate a DataLoader.

        Parameters
        ----------
        batch_size : int, optional
            How many samples per batch to load. Default = ``32``.
        shuffle : bool, optional
            Whether to shuffle the data at every epoch. Default = ``False``.
        num_workers : int, optional
            How many subprocesses to use for data loading. ``0`` means that the
            data will be loaded in the main process. Default: ``0``.
        pin_memory : bool, optional
            If ``True``, the data loader will copy Tensors into CUDA pinned
            memory before returning them.
        drop_last : bool, optional
            Set to ``True`` to drop the last incomplete batch, if the dataset
            size is not divisible by the batch size. If ``False`` and the size
            of dataset is not divisible by the batch size, then the last batch
            will be smaller. Default = ``False``.

        Returns
        -------
        _ : DataLoader
        """
        shuffle = shuffle if shuffle else self.shuffle
        shuffle &= not isinstance(dataset, IterableDataset)
        return DataLoader(
            dataset=dataset,
            batch_size=batch_size if batch_size else self.batch_size,
            shuffle=shuffle,
            num_workers=num_workers if num_workers else self.num_workers,
            pin_memory=pin_memory if pin_memory else self.pin_memory,
            drop_last=drop_last if drop_last else self.drop_last,
        )

    def default_transforms(self, stage="fit"):
        return None

    def default_preprocessing_transforms(self, **kwargs: Any) -> List[tio.Transform]:
        return None

    def default_augmentation_transforms(self, **kwargs: Any) -> List[tio.Transform]:
        return None

    def save(self) -> None:
        pass


class file_type(object):
    def __init__(self, object_type, path):
        self.type = object_type
        self.file_typ = "directory"
        self.extention = "."
        self.dcm_dir = None
        self.path = path
        self._determine_file_type()

    def _determine_file_type(self):
        data_extentions = [".json", ".xlsx", ".csv"]
        other_extentions = [".pdf", ".txt", ".mat", ".xml"]
        if self.type == "file":
            self.extention = os.path.splitext(self.path)[-1]
            try:
                if self.extention in data_extentions:
                    self.file_typ = "data"
                elif self.extention in other_extentions:
                    self.file_typ = "other_data"
                else:
                    tio.ScalarImage(self.path)
                    self.file_typ = "image"
                    if self.extention == ".dcm":
                        self.path = os.path.basename(self.path)
            except:
                self.file_typ = "other"


if __name__ == "__main__":
    test_mod = DepDataModule(
        root="/home/wangl15@acct.upmchs.net/Desktop/Raw_korean",
        base_csv="/home/antonija/Desktop/circuits/scripts/cnn_depression_linghai/data/data_Dec_8_2021.csv",
    )
    test_mod.prepare_data()
    test_mod.setup()
    print(test_mod.train_dataset[0])
