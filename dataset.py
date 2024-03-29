import os
import os.path
import cv2
import numpy as np
import h5py

from torch.utils.data import Dataset

from IPython.core import debugger
debug = debugger.Pdb().set_trace


IMG_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm']

def load_image_label_list_from_npy(img_name_list, data_root):
    cls_labels_dict = np.load('{}/cls_labels.npy'.format(data_root), allow_pickle=True).item()
    return [cls_labels_dict[img_name] for img_name in img_name_list]

class Sem_ContourData(Dataset):
    def __init__(self, split='train', data_root=None, data_list=None, transform=None , path = 'SegmentationClassAug',path_contour='contour'):
        self.split = split
        self.indices = open('{}/ImageSets/SegmentationAug/{}'.format(data_root, data_list),'r').read().splitlines()
        self.img_names = ['{}/JPEGImages/{}.jpg'.format(data_root, i) for i in self.indices]
        self.lab_names = ['{}/{}/{}.png'.format(data_root, path, i) for i in self.indices]
        self.contour_names = ['{}/{}/{}.png'.format(data_root, path_contour, i) for i in self.indices]
        self.transform = transform

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, index):
        #image_path, label_path = self.data_list[index]
        image_path=self.img_names[index]
        label_path=self.lab_names[index]
        contour_path=self.contour_names[index]
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)  # BGR 3 channel ndarray wiht shape H * W * 3
        try:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # convert cv2 read image from BGR order to RGB order
        except cv2.error:
            print(image_path)
            exit()
        image = np.float32(image)
        label = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)  # GRAY 1 channel ndarray with shape H * W
        contour = cv2.imread(contour_path, cv2.IMREAD_GRAYSCALE)
        if image.shape[0] != label.shape[0] or image.shape[1] != label.shape[1] or image.shape[1] != contour.shape[1]:
            raise (RuntimeError("Image & label shape mismatch: " + image_path + " " + label_path + "\n"))
        if self.transform is not None:
            image, label, contour = self.transform(image, label, contour)
        return image, label, contour, image_path


class Sem_ContourData_L2(Dataset):
    def __init__(self, split='train', data_root=None, data_list=None, transform=None , path = 'SegmentationClassAug',path_contour='contour',superpiex2_path = 'None',Laplace_martix_path = 'None'):
        self.split = split
        self.indices = open('{}/ImageSets/SegmentationAug/{}'.format(data_root, data_list),'r').read().splitlines()
        self.img_names = ['{}/JPEGImages/{}.jpg'.format(data_root, i) for i in self.indices]
        self.lab_names = ['{}/{}/{}.png'.format(data_root, path, i) for i in self.indices]
        self.contour_names = ['{}/{}/{}.png'.format(data_root, path_contour, i) for i in self.indices]
        self.transform = transform
        if superpiex2_path != 'None':
            self.Laplace_names = ['{}/{}/{}.mat'.format(data_root, Laplace_martix_path, i) for i in self.indices]
            self.superpiex2_names = ['{}/{}/{}.mat'.format(data_root, superpiex2_path, i) for i in self.indices]

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, index):
        #image_path, label_path = self.data_list[index]
        image_path=self.img_names[index]
        label_path=self.lab_names[index]
        contour_path=self.contour_names[index]
        Laplace_names = self.Laplace_names[index]
        superpiex2_names = self.superpiex2_names[index]

        image = cv2.imread(image_path, cv2.IMREAD_COLOR)  # BGR 3 channel ndarray wiht shape H * W * 3
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # convert cv2 read image from BGR order to RGB order
        image = np.float32(image)
        label = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)  # GRAY 1 channel ndarray with shape H * W
        contour = cv2.imread(contour_path, cv2.IMREAD_GRAYSCALE)
        Laplace = h5py.File(Laplace_names,'r')
        spiex = h5py.File(superpiex2_names,'r')
        Laplace_L = np.transpose(Laplace['L'])
        Laplace_W = np.transpose(Laplace['W'])
        spiex = np.transpose(spiex['var1'])
        if image.shape[0] != label.shape[0] or image.shape[1] != label.shape[1] or image.shape[1] != contour.shape[1] or image.shape[1] != spiex.shape[1]or image.shape[0] != spiex.shape[0]:
            print(image.shape)
            print(spiex.shape)
            raise (RuntimeError("Image & label shape mismatch: " + image_path + " " + label_path + "\n"))
        if self.transform is not None:
            image, label, contour, spiex = self.transform(image, label, contour,spiex)
        Laplace_L2 = np.zeros([1200,1200])
        [x1,y1] = Laplace_L.shape
        Laplace_L2[:x1,:y1] +=Laplace_L

        Laplace_W2 = np.zeros([1200,1200])
        [x1,y1] = Laplace_W.shape
        Laplace_W2[:x1,:y1] +=Laplace_W
        return image, label, contour, image_path, Laplace_W2, Laplace_L2, spiex

class SemData(Dataset):
    def __init__(self, split='train', data_root=None, data_list=None, transform=None , path = 'SegmentationClassAug'):
        self.split = split
        self.indices = open('{}/ImageSets/SegmentationAug/{}'.format(data_root, data_list),'r').read().splitlines()
        self.img_names = ['{}/JPEGImages/{}.jpg'.format(data_root, i) for i in self.indices]
        self.lab_names = ['{}/{}/{}.png'.format(data_root, path, i) for i in self.indices]
        self.transform = transform

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, index):
        #image_path, label_path = self.data_list[index]
        image_path=self.img_names[index]
        label_path=self.lab_names[index]
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)  # BGR 3 channel ndarray wiht shape H * W * 3
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # convert cv2 read image from BGR order to RGB order
        image = np.float32(image)
        label = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)  # GRAY 1 channel ndarray with shape H * W
        if image.shape[0] != label.shape[0] or image.shape[1] != label.shape[1]:
            raise (RuntimeError("Image & label shape mismatch: " + image_path + " " + label_path + "\n"))
        if self.transform is not None:
            image, label = self.transform(image, label)
        return image, label, image_path
        
        
class ClsData(Dataset):
    def __init__(self, split='train', data_root=None, data_list=None, transform=None , path = 'SegmentationClassAug'):
        self.split = split
        self.indices = open('{}/ImageSets/SegmentationAug/{}'.format(data_root, data_list),'r').read().splitlines()
        self.img_names = ['{}/JPEGImages/{}.jpg'.format(data_root, i) for i in self.indices]
        self.lab_names = [i for i in self.indices]
        self.lab_names2 = ['{}/{}/{}.png'.format(data_root, path, i) for i in self.indices]
        self.label_list = load_image_label_list_from_npy(self.lab_names, data_root)
        self.transform = transform

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, index):
        #image_path, label_path = self.data_list[index]
        image_path = self.img_names[index]
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)  # BGR 3 channel ndarray wiht shape H * W * 3
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # convert cv2 read image from BGR order to RGB order
        image = np.float32(image)
        label = self.label_list[index]
        label_path=self.lab_names2[index]
        label2 = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)
        if self.transform is not None:
            image,label2 = self.transform(image,label2)
        return image, label, image_path
