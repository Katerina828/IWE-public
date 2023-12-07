
import torch
import torchvision
import torchvision.transforms as transforms
from .configs import DATASET
from sklearn.model_selection import train_test_split
import numpy as np
from PIL import Image

import os
import os.path
from typing import Any, Callable, List, Optional, Tuple, Union

from PIL import Image

from torchvision.datasets.utils import  verify_str_arg
from torchvision.datasets import VisionDataset

def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')
    
class Caltech101(VisionDataset):
    """`Caltech 101 <https://data.caltech.edu/records/20086>`_ Dataset.

    .. warning::

        This class needs `scipy <https://docs.scipy.org/doc/>`_ to load target files from `.mat` format.

    Args:
        root (string): Root directory of dataset where directory
            ``caltech101`` exists or will be saved to if download is set to True.
        target_type (string or list, optional): Type of target to use, ``category`` or
            ``annotation``. Can also be a list to output a tuple with all specified
            target types.  ``category`` represents the target class, and
            ``annotation`` is a list of points from a hand-generated outline.
            Defaults to ``category``.
        transform (callable, optional): A function/transform that takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.
    """

    def __init__(
        self,
        root: str,
        target_type: Union[List[str], str] = "category",
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        download: bool = False,
    ) -> None:
        super().__init__(os.path.join(root, "caltech101"), transform=transform, target_transform=target_transform)
        os.makedirs(self.root, exist_ok=True)
        if isinstance(target_type, str):
            target_type = [target_type]
        self.target_type = [verify_str_arg(t, "target_type", ("category", "annotation")) for t in target_type]


        if not self._check_integrity():
            raise RuntimeError("Dataset not found or corrupted. You can use download=True to download it")

        self.categories = sorted(os.listdir(os.path.join(self.root, "101_ObjectCategories")))
        self.categories.remove("BACKGROUND_Google")  # this is not a real class

        # For some reason, the category names in "101_ObjectCategories" and
        # "Annotations" do not always match. This is a manual map between the
        # two. Defaults to using same name, since most names are fine.
        name_map = {
            "Faces": "Faces_2",
            "Faces_easy": "Faces_3",
            "Motorbikes": "Motorbikes_16",
            "airplanes": "Airplanes_Side_2",
        }
        self.annotation_categories = list(map(lambda x: name_map[x] if x in name_map else x, self.categories))

        self.index: List[int] = []
        self.y = []
        for (i, c) in enumerate(self.categories):
            n = len(os.listdir(os.path.join(self.root, "101_ObjectCategories", c)))
            self.index.extend(range(1, n + 1))
            self.y.extend(n * [i])

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where the type of target specified by target_type.
        """
        import scipy.io

        img = pil_loader(
            os.path.join(
                self.root,
                "101_ObjectCategories",
                self.categories[self.y[index]],
                f"image_{self.index[index]:04d}.jpg",
            )
        )

        target: Any = []
        for t in self.target_type:
            if t == "category":
                target.append(self.y[index])
            elif t == "annotation":
                data = scipy.io.loadmat(
                    os.path.join(
                        self.root,
                        "Annotations",
                        self.annotation_categories[self.y[index]],
                        f"annotation_{self.index[index]:04d}.mat",
                    )
                )
                target.append(data["obj_contour"])
        target = tuple(target) if len(target) > 1 else target[0]

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def _check_integrity(self) -> bool:
        # can be more robust and check hash of files
        return os.path.exists(os.path.join(self.root, "101_ObjectCategories"))

    def __len__(self) -> int:
        return len(self.index)


    def extra_repr(self) -> str:
        return "Target type: {target_type}".format(**self.__dict__)


def load_caltech101(train_pct:float = 0.8, steal_pct: float = 0.2, batch_size: int = 64, img_size: int = 224,label_list: list = None):

    transform = transforms.Compose(
    [
     #transforms.Resize((img_size, img_size)),
     transforms.Resize(256),
     transforms.CenterCrop(224),
     #transforms.Grayscale(num_output_channels=3),
     transforms.ToTensor(),
     transforms.Normalize(mean=[0.485, 0.456, 0.406],
                          std=[0.229, 0.224, 0.225])])
    
    caltech101_dataset = Caltech101(root=DATASET.CALTECH_ROOT, download=False, transform=transform)
    train_size = int(train_pct * len(caltech101_dataset))
    #val_size = int(val_pct * len(caltech101_dataset))
    test_size = len(caltech101_dataset) - train_size

    # Split the dataset into train and test sets
    train_set, test_set = torch.utils.data.random_split(
        caltech101_dataset,
        lengths=[
            train_size,
            #val_size,
            test_size,
        ],
        generator=torch.Generator().manual_seed(3)
    )
    
    if label_list is not None:
        # Filter the dataset to include only the images with the selected label indices
        train_set = [item for item in caltech101_dataset if item[1] in label_list]
    
    steal_size = int(steal_pct * len(train_set))

    steal_set, _ = torch.utils.data.random_split(
        train_set,
        lengths=[
            steal_size,
            len(train_set)-steal_size,
        ],
        generator=torch.Generator().manual_seed(3)
    )

    # Create data loaders for train and test sets
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)
    #val_loader = torch.utils.data.DataLoader(val_set, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False)
    steal_loader = torch.utils.data.DataLoader(steal_set, batch_size=batch_size, shuffle=True)

    return train_loader,test_loader,test_loader,steal_loader
    
