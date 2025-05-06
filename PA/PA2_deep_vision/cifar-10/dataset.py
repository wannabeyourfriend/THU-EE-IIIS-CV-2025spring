import os
import os.path
import numpy as np
import pickle
import torch 
import torchvision.transforms as tfs
from PIL import Image

class CIFAR10(torch.utils.data.Dataset):
    """
        modified from `CIFAR10 <https://www.cs.toronto.edu/~kriz/cifar.html>`_ Dataset.
    """
    def __init__(self, train=True):
        super(CIFAR10, self).__init__()

        self.base_folder = '../datasets/cifar-10-batches-py'
        self.train_list = ['data_batch_1', 'data_batch_2', 'data_batch_3', 'data_batch_4','data_batch_5']
        self.test_list = ['test_batch']

        self.meta = {
            'filename': 'batches.meta',
            'key': 'label_names'
        }

        self.train = train  # training set or test set
        if self.train:
            file_list = self.train_list
        else:
            file_list = self.test_list

        self.data = []
        self.targets = []

        # now load the picked numpy arrays
        for file_name in file_list:
            file_path = os.path.join(self.base_folder, file_name)
            with open(file_path, 'rb') as f:
                entry = pickle.load(f, encoding='latin1')
                self.data.append(entry['data'])
                if 'labels' in entry:
                    self.targets.extend(entry['labels'])
                else:
                    self.targets.extend(entry['fine_labels'])

        self.data = np.vstack(self.data).reshape(-1, 3, 32, 32)
        self.data = self.data.transpose((0, 2, 3, 1))  # convert to HWC

        self._load_meta()

    def _load_meta(self):
        path = os.path.join(self.base_folder, self.meta['filename'])
        with open(path, 'rb') as infile:
            data = pickle.load(infile, encoding='latin1')
            self.classes = data[self.meta['key']]
        self.class_to_idx = {_class: i for i, _class in enumerate(self.classes)}

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.targets[index]
        img = img.astype(np.float32)
        img = img.transpose(2, 0, 1)
        
        # ------------TODO--------------
        # data augmentation
        # ------------TODO--------------
        # Convert to tensor
        img_tensor = torch.from_numpy(img)
        
        # Apply data augmentation only for training set
        if self.train:
            # Random horizontal flip with 0.5 probability
            if np.random.random() > 0.5:
                img_tensor = torch.flip(img_tensor, dims=[2])  # Flip horizontally
                
            # Random crop with padding
            padding = 4
            padded = torch.nn.functional.pad(img_tensor, (padding, padding, padding, padding), mode='reflect')
            crop_h = np.random.randint(0, 2 * padding)
            crop_w = np.random.randint(0, 2 * padding)
            img_tensor = padded[:, crop_h:crop_h + 32, crop_w:crop_w + 32]
            
            # 颜色增强 - 随机调整亮度
            if np.random.random() > 0.5:
                brightness_factor = np.random.uniform(0.8, 1.2)
                img_tensor = img_tensor * brightness_factor
                img_tensor = torch.clamp(img_tensor, 0, 255)
            
            # Normalize
            img_tensor = img_tensor / 255.0
        else:
            # Just normalize for validation set
            img_tensor = img_tensor / 255.0

        return img_tensor, target

    def __len__(self):
        return len(self.data)


if __name__ == '__main__':
    # --------------------------------
    # The resolution of CIFAR-10 is tooooo low
    # You can use Lenna.png as an example to visualize and check your code.
    # Submit the origin image "Lenna.png" as well as at least two augmented images of Lenna named "Lenna_aug1.png", "Lenna_aug2.png" ...
    # --------------------------------

    # # Visualize CIFAR-10. For someone who are intersted.
    # train_dataset = CIFAR10()
    # i = 0
    # for imgs, labels in train_dataset:
    #     imgs = imgs.transpose(1,2,0)
    #     cv2.imwrite(f'aug1_{i}.png', imgs)
    #     i += 1
    #     if i == 10:
    #         break 

    # Visualize and save for submission
    img = Image.open('Lenna.png')
    img.save('../results/Lenna.png')

    # --------------TODO------------------
    # Copy the first kind of your augmentation code here
    # --------------TODO------------------
    # Horizontal flip
    aug1 = img.transpose(Image.FLIP_LEFT_RIGHT)
    aug1.save(f'../results/Lenna_aug1.png')

    # --------------TODO------------------
    # Copy the second kind of your augmentation code here
    # --------------TODO------------------
    # 颜色增强 - 调整亮度和对比度
    from PIL import ImageEnhance
    
    # 增强亮度
    brightness_enhancer = ImageEnhance.Brightness(img)
    aug2 = brightness_enhancer.enhance(1.5)  # 亮度提高50%
    
    # 增强对比度
    contrast_enhancer = ImageEnhance.Contrast(aug2)
    aug2 = contrast_enhancer.enhance(1.3)  # 对比度提高30%
    
    aug2.save(f'../results/Lenna_aug2.png')

