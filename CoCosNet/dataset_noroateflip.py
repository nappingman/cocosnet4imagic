import torch
import numpy as np
import cv2 as cv

from PIL import Image
from torch.utils.data import Dataset
from pathlib import Path
from typing import List
from typing_extensions import Literal

from thin_plate_spline import warping_image
from torchvision.transforms import ColorJitter
from hint_processor import LineProcessor

LineArt = List[Literal["xdog", "pencil", "digital", "blend"]]


class IllustDataset(Dataset):
    """Dataset for training.

       Returns (line, color)
           line: input. Line art of color image
           color: target.
    """
    def __init__(self,
                 data_path: Path,
                 sketch_path: Path,
                 line_method: LineArt,
                 extension=".jpg",
                 train_size=224,
                 valid_size=256,
                 color_space="rgb",
                 line_space="rgb",
                 src_perturbation=0.5,
                 tgt_perturbation=0.2):

        self.data_path = data_path
        self.pathlist = list(self.data_path.glob(f"**/*{extension}"))
        self.train_list, self.val_list = self._train_val_split(self.pathlist)
        self.train_len = len(self.train_list)

        self.train_size = train_size
        self.valid_size = valid_size

        self.line_process = LineProcessor(sketch_path, line_method)
        self.color_space = color_space
        self.line_space = line_space

        self.sketch_path = sketch_path
        self.src_per = src_perturbation
        self.tgt_per = tgt_perturbation
        self.thre = 50

        self.src_const = np.array([
            [-0.5, -0.5],
            [0.5, -0.5],
            [-0.5, 0.5],
            [0.5, 0.5],
            [0.2, -0.2],
            [-0.2, 0.2],
            [0.2, 0.2],
            [-0.2, -0.2]
        ])

    @staticmethod
    def _train_val_split(pathlist: List) -> (List, List):
        split_point = int(len(pathlist) * 0.995)
        train = pathlist[:split_point]
        val = pathlist[split_point:]

        return train, val

    @staticmethod
    def _coordinate(img: np.array,
                    color_space: str) -> np.array:
        if color_space == "yuv":
            img = img.astype(np.uint8)
            img = cv.cvtColor(img, cv.COLOR_BGR2YCrCb)
            img = img.transpose(2, 0, 1).astype(np.float32)
            img = (img - 127.5) / 127.5
        elif color_space == "gray":
            img = img.astype(np.uint8)
            img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
            img = np.expand_dims(img, axis=0).astype(np.float32)
            img = (img - 127.5) / 127.5
        else:
            img = img[:, :, ::-1].astype(np.float32)
            img = (img.transpose(2, 0, 1) - 127.5) / 127.5

        return img

    @staticmethod
    def _totensor(array_list: List) -> torch.Tensor:
        return torch.cuda.FloatTensor(np.array(array_list).astype(np.float32))

    @staticmethod
    def _random_crop(line: np.array,
                     color: np.array,
                     size: int) -> (np.array, np.array):
        scale = np.random.randint(396, 512)
        line = cv.resize(line, (scale, scale))
        color = cv.resize(color, (scale, scale))

        height, width = line.shape[0], line.shape[1]
        rnd0 = np.random.randint(height - size - 1)
        rnd1 = np.random.randint(width - size - 1)

        line = line[rnd0: rnd0 + size, rnd1: rnd1 + size]
        color = color[rnd0: rnd0 + size, rnd1: rnd1 + size]

        return line, color

    def _warp(self, img,img2):
        const = self.src_const
        c_src = const + np.random.uniform(-self.src_per, self.src_per, (8, 2))
        c_tgt = c_src + np.random.uniform(-self.tgt_per, self.tgt_per, (8, 2))

        img = warping_image(img, c_src, c_tgt)
        img2 = warping_image(img2, c_src, c_tgt)

        return img,img2

    def _jitter(self, img):
        img = img.astype(np.float32)
        noise = np.random.uniform(-self.thre, self.thre)
        img += noise
        img = np.clip(img, 0, 255)

        return img
        
    def rotate(self, img,img2):
        rows, cols, channel = img.shape
        degree = np.random.randint(0,180)
        
        M = cv.getRotationMatrix2D((cols/2, rows/2), degree, 1)
        rotated = cv.warpAffine(img, M, (cols, rows))
        rotated2 = cv.warpAffine(img2, M, (cols, rows))
        return rotated, rotated2
    def flip(self, img,img2):
        if np.random.choice([0,1],p=[0.5,0.5])==1:
            img = cv.flip(img, 0)
            img2 = cv.flip(img2, 0)
        if np.random.choice([0,1],p=[0.5,0.5])==1:
            img = cv.flip(img, 1)
            img2 = cv.flip(img2, 1)
        
       

        return img,img2

    def _preprocess(self, color, line):
        """3 stages of preparation
           - Crop
           - Spatial & Color augmentation
           - Coordination
        """
        line, color = self._random_crop(line, color, size=self.train_size)

        jittered = self._jitter(color)
        warped,warped_line = self._warp(jittered,line)
        #warped,warped_line = self.rotate(warped,warped_line)
        #warped,warped_line = self.flip(warped,warped_line)
        

        jittered = self._coordinate(jittered, self.color_space)
        warped = self._coordinate(warped, self.color_space)
        line = self._coordinate(line, self.line_space)
        warped_line = self._coordinate(warped_line, self.line_space)

        return jittered, warped, line,warped_line

    def valid(self, validsize):
        c_valid_box = []
        l_valid_box = []

        for index in range(validsize):
            color_path = self.val_list[index]
            color = cv.imread(str(color_path))
            line = self.line_process(color_path)

            jitter = self._jitter(color)
            warp = self._warp(jitter)

            color = self._coordinate(warp, self.color_space)
            line = self._coordinate(line, self.line_space)

            c_valid_box.append(color)
            l_valid_box.append(line)

        color = self._totensor(c_valid_box)
        line = self._totensor(l_valid_box)

        return color, line

    def __repr__(self):
        return f"dataset length: {self.train_len}"

    def __len__(self):
        return self.train_len

    def __getitem__(self, idx):
        # Color prepare
        color_path = self.train_list[idx]
        color = cv.imread(str(color_path))

        # Line prepare
        line = self.line_process(color_path)
        jit, war, line,warped_line = self._preprocess(color, line)
        
        war = self._totensor(war)
        line = self._totensor(line)
        warped_line = self._totensor(warped_line)
        jit = self._totensor(jit)
        
        input_dict = {'label': line,
                      'image': jit,
                      'path': str(color_path),
                      'self_ref': torch.ones_like(war),
                      'ref': war,
                      'label_ref': warped_line
                      }

        return input_dict

class IllustTestDataset(Dataset):
    """Dataset for training.

       Returns (line, color)
           line: input. Line art of color image
           color: target.
    """
    def __init__(self,
                 data_path: Path,
                 sketch_path: Path,
                 line_method: LineArt,
                 extension=".jpg",
                 valid_size=256):
        self.valid_size =valid_size
        self.data_path = data_path
        self.pathlist = list(self.data_path.glob(f"**/*{extension}"))
        self.color_space = "rgb"
        self.line_space = "rgb"
        self.line_process = LineProcessor(sketch_path, line_method)
        self.train_len = len(self.pathlist)
        self.src_per = 0.5
        self.tgt_per = 0.2
        self.thre = 50

        self.src_const = np.array([
            [-0.5, -0.5],
            [0.5, -0.5],
            [-0.5, 0.5],
            [0.5, 0.5],
            [0.2, -0.2],
            [-0.2, 0.2],
            [0.2, 0.2],
            [-0.2, -0.2]
        ])

    @staticmethod
    def _coordinate(img: np.array,
                    color_space: str) -> np.array:
        if color_space == "yuv":
            img = img.astype(np.uint8)
            img = cv.cvtColor(img, cv.COLOR_BGR2YCrCb)
            img = img.transpose(2, 0, 1).astype(np.float32)
            img = (img - 127.5) / 127.5
        elif color_space == "gray":
            img = img.astype(np.uint8)
            img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
            img = np.expand_dims(img, axis=0).astype(np.float32)
            img = (img - 127.5) / 127.5
        else:
            img = img[:, :, ::-1].astype(np.float32)
            img = (img.transpose(2, 0, 1) - 127.5) / 127.5

        return img

    @staticmethod
    def _totensor(array_list: List) -> torch.Tensor:
        return torch.cuda.FloatTensor(np.array(array_list).astype(np.float32))

    def _jitter(self, img):
        img = img.astype(np.float32)
        noise = np.random.uniform(-self.thre, self.thre)
        img += noise
        img = np.clip(img, 0, 255)

        return img
    def _warp(self, img,img2):
        const = self.src_const
        c_src = const + np.random.uniform(-self.src_per, self.src_per, (8, 2))
        c_tgt = c_src + np.random.uniform(-self.tgt_per, self.tgt_per, (8, 2))

        img = warping_image(img, c_src, c_tgt)
        img2 = warping_image(img2, c_src, c_tgt)

        return img,img2
    def _preprocess(self, color, line):
        """3 stages of preparation
           - Crop
           - Spatial & Color augmentation
           - Coordination
        """
        line, color = self._random_crop(line, color, size=self.train_size)

        jittered = self._jitter(color)
        warped,warped_line = self._warp(jittered,line)

        jittered = self._coordinate(jittered, self.color_space)
        warped = self._coordinate(warped, self.color_space)
        line = self._coordinate(line, self.line_space)
        warped_line = self._coordinate(warped_line, self.line_space)

        return jittered, warped, line,warped_line

    def __repr__(self):
        return f"dataset length: {self.train_len}"

    def __len__(self):
        return self.train_len

    def __getitem__(self, idx):
        # Color prepare
        color_path = self.pathlist[idx]
        color = cv.imread(str(color_path))
        
        color = cv.resize(color,(self.valid_size,self.valid_size))
        
        

        # Line prepare
        line = self.line_process(color_path)
        line = cv.resize(line,(self.valid_size,self.valid_size))
        
        warped,warped_line = self._warp(color,line)
        
        
        color = self._coordinate(color, self.color_space)
        line = self._coordinate(line, self.line_space)
        
        warped = self._coordinate(warped, self.color_space)
        warped_line = self._coordinate(warped_line, self.line_space)

        
        warped = self._totensor(warped)
        warped_line = self._totensor(warped_line)
        
        line = self._totensor(line)
        color = self._totensor(color)
        
        input_dict = {'label': line,
                      'image': color,
                      'path': str(color_path),
                      'self_ref': torch.ones_like(color),
                      'ref': warped,
                      'label_ref': warped_line
                      }

        return input_dict
# class IllustTestDataset(Dataset):
    # """Dataset for inference/test.

       # Returns (line_path, color_path)
           # line_path: path of line art
           # color_path: path of color image
    # """
    # def __init__(self,
                 # data_path: Path,
                 # sketch_path: Path):

        # self.path = data_path
        # self.pathlist = list(self.path.glob('**/*.png'))
        # self.pathlen = len(self.pathlist)

        # self.sketch_path = sketch_path
        # self.test_len = 200

    # def __repr__(self):
        # return f"dataset length: {self.pathlen}"

    # def __len__(self):
        # return self.test_len

    # def __getitem__(self, idx):
        # line_path = self.pathlist[idx]
        # line_path = self.sketch_path / line_path.name

        # rnd = np.random.randint(self.pathlen)
        # style_path = self.pathlist[rnd]

        # return line_path, style_path


class LineCollator:
    """Collator for training.
    """
    def __init__(self,
                 img_size=224,
                 src_perturbation=0.5,
                 dst_perturbation=0.2,
                 brightness=0.3,
                 contrast=0.5,
                 saturation=0.1,
                 hue=0.3):

        self.size = img_size
        self.src_per = src_perturbation
        self.tgt_per = dst_perturbation
        self.thre = 50

        self.src_const = np.array([
            [-0.5, -0.5],
            [0.5, -0.5],
            [-0.5, 0.5],
            [0.5, 0.5],
            [0.2, -0.2],
            [-0.2, 0.2],
            [0.2, 0.2],
            [-0.2, -0.2]
        ])

        self.jittering = ColorJitter(brightness, contrast, saturation, hue)

    @staticmethod
    def _random_crop(line, color, size):
        height, width = line.shape[0], line.shape[1]
        rnd0 = np.random.randint(height - size - 1)
        rnd1 = np.random.randint(width - size - 1)

        line = line[rnd0: rnd0 + size, rnd1: rnd1 + size]
        color = color[rnd0: rnd0 + size, rnd1: rnd1 + size]

        return line, color

    @staticmethod
    def _coordinate(image):
        """3 stage of manipulation
           - BGR -> RGB
           - (H, W, C) -> (C, H, W)
           - Normalize
        
        Parameters
        ----------
        image : numpy.array
            image data
        
        Returns
        -------
        numpy.array
            manipulated image data
        """
        image = image[:, :, ::-1]
        image = image.transpose(2, 0, 1)
        image = (image - 127.5) / 127.5

        return image

    @staticmethod
    def _totensor(array_list):
        array = np.array(array_list).astype(np.float32)
        tensor = torch.FloatTensor(array)
        tensor = tensor.cuda()

        return tensor

    def _warp(self, img):
        """Spatial augment by TPS
        """
        const = self.src_const
        c_src = const + np.random.uniform(-self.src_per, self.src_per, (8, 2))
        c_tgt = c_src + np.random.uniform(-self.tgt_per, self.tgt_per, (8, 2))

        img = warping_image(img, c_src, c_tgt)

        return img

    def _jitter(self, img):
        """Color augment
        """
        img = img.astype(np.float32)
        noise = np.random.uniform(-self.thre, self.thre)
        img += noise
        img = np.clip(img, 0, 255)

        return img

    def _prepair(self, color, line):
        """3 stages of preparation
           - Crop
           - Spatial & Color augmentation
           - Coordination
        """
        line, color = self._random_crop(line, color, size=self.size)

        jittered = self._jitter(color)
        warped = self._warp(jittered)

        jittered = self._coordinate(jittered)
        warped = self._coordinate(warped)
        line = self._coordinate(line)

        return jittered, warped, line

    def __call__(self, batch):
        j_box = []
        w_box = []
        l_box = []

        for b in batch:
            color, line = b
            jitter, warped, line = self._prepair(color, line)

            j_box.append(jitter)
            w_box.append(warped)
            l_box.append(line)

        j = self._totensor(j_box)
        w = self._totensor(w_box)
        l = self._totensor(l_box)

        return (j, w, l)


class LineTestCollator:
    """Collator for inference/test.
    """
    def __init__(self):
        pass

    @staticmethod
    def _coordinate(image):
        image = image[:, :, ::-1]
        image = image.transpose(2, 0, 1)
        image = (image - 127.5) / 127.5

        return image

    @staticmethod
    def _totensor(array_list):
        array = np.array(array_list).astype(np.float32)
        tensor = torch.FloatTensor(array)
        tensor = tensor.cuda()

        return tensor

    def _prepare(self, image_path, style_path, size=512):
        line = cv.imread(str(image_path))
        line = cv.resize(line, (512, 512), interpolation=cv.INTER_CUBIC)
        color = cv.imread(str(style_path))

        color = self._coordinate(color)
        line = self._coordinate(line)

        return color, line

    def __call__(self, batch):
        c_box = []
        l_box = []

        for bpath, style in batch:
            color, line = self._prepare(bpath, style)

            c_box.append(color)
            l_box.append(line)

        c = self._totensor(c_box)
        l = self._totensor(l_box)

        return (c, l)