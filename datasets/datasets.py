import csv
import copy
from typing import List, Tuple

from osgeo import gdal
import numpy as np
import torch
from torch.utils.data import Dataset

from albumentations import RandomCrop, Compose, Normalize

class SN8Dataset(Dataset):
    def __init__(self,
                 csv_filename: str,
                 data_to_load: List[str] = ["preimg","postimg","building","road","roadspeed","flood"],
                 img_size: Tuple[int, int] = (1300,1300),
                 transforms=None,
                 crop=None):
        """ pytorch dataset for spacenet-8 data. loads images from a csv that contains filepaths to the images
        
        Parameters:
        ------------
        csv_filename (str): absolute filepath to the csv to load images from. the csv should have columns: preimg, postimg, building, road, roadspeed, flood.
            preimg column contains filepaths to the pre-event image tiles (.tif)
            postimg column contains filepaths to the post-event image tiles (.tif)
            building column contains the filepaths to the binary building labels (.tif)
            road column contains the filepaths to the binary road labels (.tif)
            roadspeed column contains the filepaths to the road speed labels (.tif)
            flood column contains the filepaths to the flood labels (.tif)
        data_to_load (list): a list that defines which of the images and labels to load from the .csv. 
        img_size (tuple): the size of the input pre-event image in number of pixels before any augmentation occurs.
        
        """
        self.all_data_types = ["preimg", "postimg", "building", "road", "roadspeed", "flood"]
        
        self.img_size = img_size
        self.data_to_load = data_to_load
        
        self.files = []

        dict_template = {}
        for i in self.all_data_types:
            dict_template[i] = None
        
        with open(csv_filename, newline='') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                in_data = copy.copy(dict_template)
                for j in self.data_to_load:
                    in_data[j]=row[j]
                self.files.append(in_data)
        
        if crop:
           self.crop = self.get_crop_transforms(crop, data_to_load)
        else:
            self.crop = None

        self.transforms = transforms
        print("loaded", len(self.files), "image filepaths")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        data_dict = self.files[index]

        returned_data = {}
        for i in self.all_data_types:
            filepath = data_dict[i]
            if filepath:
                # need to resample postimg to same spatial resolution/extent as preimg and labels.
                if i == "postimg":
                    ds = self.get_warped_ds(data_dict["postimg"])
                else:
                    ds = gdal.Open(filepath)
                image = ds.ReadAsArray()
                ds = None
                # if len(image.shape)==2: # add a channel axis if read image is only shape (H,W).
                #     returned_data.append(torch.unsqueeze(torch.from_numpy(image), dim=0).float())
                # else:
                #     returned_data.append(torch.from_numpy(image).float())
                
                if len(image.shape)==2: 
                    image = np.expand_dims(image, axis=0)                
                returned_data[i] = image.transpose()
            
            else:
                returned_data[i] = None

        if self.crop:
            returned_data = self.crop_transform(returned_data)
        
        for key in self.all_data_types: 
            if returned_data[key] is not None:
                returned_data[key] = torch.from_numpy(returned_data[key]).float().permute(2,0,1)
            else:
                returned_data[key] = 0


        out = [returned_data["preimg"], 
                    returned_data["postimg"],
                    returned_data["building"],
                    returned_data["road"],
                    returned_data["roadspeed"],
                    returned_data["flood"]]            
        return out

    def get_image_filename(self, index: int) -> str:
        """ return pre-event image absolute filepath at index """
        data_dict = self.files[index]
        return data_dict["preimg"]

    def get_warped_ds(self, post_image_filename: str) -> gdal.Dataset:
        """ gdal warps (resamples) the post-event image to the same spatial resolution as the pre-event image and masks 
        
        SN8 labels are created from referencing pre-event image. Spatial resolution of the post-event image does not match the spatial resolution of the pre-event imagery and therefore the labels.
        In order to align the post-event image with the pre-event image and mask labels, we must resample the post-event image to the resolution of the pre-event image. Also need to make sure
        the post-event image covers the exact same spatial extent as the pre-event image. this is taken care of in the the tiling"""
        ds = gdal.Warp("", post_image_filename,
                       format='MEM', width=self.img_size[1], height=self.img_size[0],
                       resampleAlg=gdal.GRIORA_Bilinear,
                       outputType=gdal.GDT_Byte)
        return ds

    def get_crop_transforms(self, crop : Tuple[int, int], data_to_load : dict):
        additional_targets = {key : "image" for key in data_to_load[1:]}
        transform = Compose(RandomCrop(*crop, always_apply=True),
                            additional_targets=additional_targets) 
        return transform

    def crop_transform(self, data : dict):
        crop_data = {}
        crop_data["image"] = data[self.data_to_load[0]]
        for key in self.data_to_load[1:]:
            crop_data[key] = data[key]
        crop_data = self.crop(**crop_data)
        data[self.data_to_load[0]] = crop_data["image"]
        for key in self.data_to_load[1:]:
            data[key] = crop_data[key]    
        return data

    


if __name__ == "__main__":

                            
    
    
    train_dataset = SN8Dataset("areas_of_interest/sn8_data_train.csv",
                            data_to_load=["preimg","postimg","flood"], # ,"building", "road", "roadspeed", "flood"],
                            img_size=(1300, 1300),
                            transforms=None,
                            crop=(512, 512))
    
    #print(train_dataset[30])
    x = torch.utils.data.DataLoader(train_dataset,batch_size=1)
    for i in x:
        y, z, s, d, f, g = i
        print(y)
    # import matplotlib.pyplot as plt

    # for x, image in enumerate(train_dataset):
            
    #     plt.imshow(image["image"])
    #     plt.savefig(f"testfiles/test{x}-pre.png")
    #     plt.imshow(image["postimg"])
    #     plt.savefig(f"testfiles/test{x}-post.png")
    #     plt.show()

    