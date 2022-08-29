import logging
from optparse import Option
import sys
import os
from treelib import Node, Tree
from pyrsistent import optional
import torch
import torch.utils.data
import torchio as tio
import monai.transforms
import radio.data as radata
from radio.settings.pathutils import is_dir_or_symlink, PathType
from radio.data.datatypes import SpatialShapeType
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union, cast
from project.misc import nifti_helpers
import pandas as pd
import numpy as np


Sample = List[Tuple[Path, Any]]
OneSample = Union[Dict[str, Tuple[Any, ...]], Tuple[Any, ...]]

# TODO: make something to compile subject ID, scan ID and target


class DepDataModule(radata.CerebroDataModule):
    """
    This class will be based on the folder dataset
    in radio for our specific project
    DO NOT use this with a queue yet
    """

    def __init__(
        self,
        *args,
        root: PathType,
        subject_dicts: Union[dict[str, list[str]], dict[str, pd.DataFrame]],
        study: str = "",
        subj_dir: str = "Public/data",
        data_dir: str = "",
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
        base_csv: pd.DataFrame = None,
        subjectCol: Optional[str] = None,
        scanCol: Optional[str] = None,
        label_col: Optional[list[str]] = None,
        transform: Optional[Callable] = None,
        **kwargs,
    ) -> None:
        super().__init__(
            *args,
            root,
            study,
            subj_dir,
            data_dir,
            modalities,
            labels,
            train_transforms,
            val_transforms,
            test_transforms,
            use_augmentation,
            use_preprocessing,
            resample,
            None,
            probability_map,
            create_custom_probability_map,
            label_name,
            label_probabilities,
            queue_max_length,
            samples_per_volume,
            batch_size,
            shuffle_subjects,
            shuffle_patches,
            num_workers,
            pin_memory,
            start_background,
            drop_last,
            num_folds,
            val_split,
            dims,
            seed,
            verbose,
            **kwargs,
        )
        # input dict should have the same keys
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

        if base_csv:
            self.data = pd.read_csv(base_csv)
        else:
            self.file_tree = self.parse_root(self.root)
            pass

    def prepare_data(self, *args: Any, **kwargs: Any) -> None:
        """
        Verify data directory exists.
        Verify if test/train/val splitted.
        """
        if not is_dir_or_symlink(self.root):
            raise OSError("Study data directory not found!")
        self.check_if_data_split()

    def setup(self, *args: Any, stage: Option[str] = None, **kwargs: Any) -> None:
        if stage is None or stage is "fit":
            train_transforms = (
                self.default_transforms(stage="fit")
                if self.train_transforms is None
                else self.train_transforms
            )
        if stage is None or stage is "val":
            val_transforms = (
                self.default_transforms(stage="fit")
                if self.val_transforms is None
                else self.val_transforms
            )
        if stage is None or stage is "fit":
            train_list = self.get_subjects(fold="train")
            self.train_dataset = tio.SubjectsDataset(
                train_list, transform=train_transforms
            )
            self.train_loader = super().dataloader(self.train_dataset)

        if stage is None or stage is "fit":
            val_list = self.get_subjects(fold="val")
            self.val_dataset = tio.SubjectsDataset(val_list, transform=val_transforms)
            self.val_loader = super().dataloader(self.val_dataset)

        if stage is None or stage is "fit":
            test_list = self.get_subjects(fold="test")
            self.test_dataset = tio.SubjectsDataset(test_list)
            self.test_loader = super().dataloader(self.test_dataset)

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
        print(combined_sub_scan[:, 0])
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

    @staticmethod
    def parse_root(root) -> Tree:
        """parses the folder structure of the root path given. designed for ABD raw folder.

        Args:
            root (str): path of the starting folder for the walk

        Returns:
            Tree: tree of files and folders in the root folder
        """
        file_tree = Tree()
        ignore_files = [".DS_Store"]
        file_tree.create_node(
            os.path.dirname(root), root, data=file_type("directory", root),
        )
        for root, dirs, files in os.walk(root):
            for name in dirs:
                file_tree.create_node(
                    name,
                    os.path.join(root, name),
                    parent=root,
                    data=file_type("directory", os.path.join(root, name)),
                )
            for name in files:
                if not name in ignore_files:
                    file_tree.create_node(
                        name,
                        os.path.join(root, name),
                        parent=root,
                        data=file_type("file", os.path.join(root, name)),
                    )

        return file_tree

    def get_subjects(self, fold: str = "train") -> List[tio.Subject]:
        train_subjs, test_subjs, val_subjs = self.get_subjects_dicts(
            input=self.inputs, labels=self.labels
        )
        if fold == "train":
            subjs_dict = train_subjs
        elif fold == "test":
            subjs_dict = test_subjs
        else:
            subjs_dict = val_subjs

    def get_subjects_lists(self, input, labels) -> List[tio.Subject]:
        def _get_subjects_list(
            data: pd.DataFrame, subject_list: pd.DataFrame
        ) -> Tuple[List[tio.Subject], List[tio.Subject], List[tio.Subject]]:
            # Need to drop na values somewhere before this
            subset = data[data["subjectFolder"].isin(subject_list["subjectFolder"])]
            data_input = subset[input].values
            data_labels = subset[labels].values
            dummy_image = tio.ScalarImage(tensor=torch.rand(1, 128, 128, 128))
            subjects = []
            for sub in zip(data_input, data_labels):
                subjects.append(
                    tio.Subject(
                        {"dummy": dummy_image, "input": sub[0], "labels": sub[1]}
                    )
                )
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
    test_mod = DepDataModule()
    test_mod.prepare_data()
    test_mod.setup()

