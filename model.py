import torch
from torch import nn,optim
import torch.nn.functional as F
import glob
from torch.utils.data import Dataset
from PIL import Image
import cv2
from pycocotools import coco
import matplotlib.pyplot as plt 
import numpy as np
import time 
from torchvision.transforms import v2
import sys

class SeqImageDataset(Dataset):
    def __init__(self, dataset_path, label_path, img_transforms=None, sequence_length=3):
        
        self.img_transforms = img_transforms
        self.sequence_length = sequence_length

        self.coco_obj = coco.COCO(label_path)
        
        self.img_ids = self.coco_obj.getImgIds()[:3844] ###until last image annotated

        self.files = sorted(glob(f'{dataset_path}/*.png'))

        self.annotated = []

        for a in range(0,len(self.img_ids)):
            anns = self.coco_obj.getAnnIds(imgIds=[self.img_ids[a]])
            if(len(anns)>0):
                 self.annotated.append(a)
     
        self.not_annotated = np.setdiff1d(np.arange(0,len(self.img_ids)),np.array(self.annotated))

        self.not_annotated = self.not_annotated[np.random.choice(len(self.not_annotated), len(self.annotated))]

        self.annotated = list(self.annotated) + list(self.not_annotated)

        print(f'Loaded {len(self.files)} images from {dataset_path} dataset')
        
    def __len__(self):
        return len(self.files) - self.sequence_length
    

    # def __getitem__(self, idx):
    #     images = [
    #         torchvision.io.read_image(self.files[idx+i], mode=torchvision.io.ImageReadMode.GRAY) for i in range(self.sequence_length)
    #     ]
    #     images = torch.stack(images, dim=0)
    #     labels = self.labels.iloc[idx : idx + self.sequence_length]            

    #     if self.img_transforms:
    #         images = self.img_transforms(images)
            
    #     return images, labels
    
    
class FrondandDiff(Dataset):
    def __init__(self, transform=None):

        self.transform = transform

        self.coco_obj = coco.COCO("instances_default.json")
        

        self.img_ids = self.coco_obj.getImgIds()[:3844]###until last image annotated

        self.annotated = []
        self.events = []

        event_list = []
        temp_list_zero = []
        temp_list_one = []

        a_id_prev = 0

        for a in range(0,len(self.img_ids)):
            anns = self.coco_obj.getAnnIds(imgIds=[self.img_ids[a]])
            

            if(len(anns)>0):
                
                attr = [self.coco_obj.loadAnns(ids=anns)[i]['attributes']['id'] for i in range(len(anns))]
                self.annotated.append(a)

                if (self.img_ids[a] - a_id_prev == 1) or (self.img_ids[a] == 0):

                    for atr_id in attr:

                        if atr_id == 0:
                            temp_list_zero.append(a)

                        elif atr_id == 1:
                            temp_list_one.append(a)

                elif self.img_ids[a] - a_id_prev > 1:

                    if len(temp_list_zero) > 0:
                        event_list.append(temp_list_zero)
                    
                    if len(temp_list_one) > 0:
                        event_list.append(temp_list_one)

                    temp_list_zero = []
                    temp_list_one = []
                
                a_id_prev = self.img_ids[a]

        self.events = event_list

        self.not_annotated = np.setdiff1d(np.arange(0,len(self.img_ids)),np.array(self.annotated))
        
        seed = 1997

        np.random.seed(seed)
        self.not_annotated = self.not_annotated[np.random.choice(len(self.not_annotated), len(self.annotated), replace=False)]

        self.annotated = list(self.annotated) + list(self.not_annotated)
        self.annotated = sorted(self.annotated)

        len_train_noevents = int(np.floor(len(self.not_annotated)*0.8))

        seed = 1997
        np.random.seed(seed)
        rand_arr_noevent = np.random.choice(int(len(self.not_annotated)), len_train_noevents, replace=False)

        len_train_events = int(np.floor(len(self.events)*0.8))

        seed = 1997
        np.random.seed(seed)
        rand_arr = np.random.choice(int(len(self.events)), len_train_events, replace=False)

        train_data = []
        test_data = []

        for i in np.arange(len(self.not_annotated)):
            
            if i in rand_arr_noevent:
                train_data.append(self.not_annotated[i])
            else:
                test_data.append(self.not_annotated[i])

        for i in np.arange(len(self.events)):
            if i in rand_arr:
                train_data.extend(self.events[i])
            else:
                test_data.extend(self.events[i])

        train_data_idx = []
        test_data_idx = []
        self.annotated = np.array(self.annotated)

        for td in train_data:
            val = np.where(np.array(self.annotated) == td)[0]
            train_data_idx.extend(val)

        for td in test_data:
            val = np.where(np.array(self.annotated) == td)[0]
            test_data_idx.extend(val)

        self.train_data_idx = sorted(train_data_idx)
        self.test_data_idx = sorted(test_data_idx)

        # if(training):
        #     self.annot_paths = self.annot_paths[:int(len(self.annot_paths)*0.7)]
        
    

    def get_img_and_annotation(self,idx):

        resize_par = 256

        # Pick one image.
        # print('idx', idx)
        # if idx > 1:
        #     idx_prev = idx-1

        #     img_id_prev   = self.annotated[idx_prev]

        #     print(idx_prev)
        #     print(idx)
        #     print(self.annotated[idx])
        #     print(self.annotated[idx_prev])
        #     img_info_prev = self.coco_obj.loadImgs([img_id_prev])[0]
        #     img_file_name_prev = img_info_prev["file_name"]

        #     im_prev = np.asarray(Image.open("/Volumes/SSD/differences/"+img_file_name_prev).convert("L"))/255.0

        #     im_prev = cv2.resize(im_prev  , (resize_par , resize_par),interpolation = cv2.INTER_CUBIC)

        #     annotations_prev = self.coco_obj.getAnnIds(imgIds=img_id_prev)

        #     if(len(annotations_prev)==0):
        #         im_prev = np.zeros(np.shape(im_prev))

        # else:
        #     im_prev = np.zeros((1024, 1024))

        img_id   = self.annotated[idx]
        img_info = self.coco_obj.loadImgs([img_id])[0]
        img_file_name = img_info["file_name"]

        # Use URL to load image.
        im = np.asarray(Image.open("/Volumes/SSD/differences/"+img_file_name).convert("L"))/255.0

        im = cv2.resize(im  , (resize_par , resize_par),interpolation = cv2.INTER_CUBIC)

        GT = np.zeros((2,1024,1024))
        annotations = self.coco_obj.getAnnIds(imgIds=img_id)

        if(len(annotations)>0):
            for a in annotations:
                ann = self.coco_obj.loadAnns(a)
                GT[int(ann[0]["attributes"]["id"]),:,:]=coco.maskUtils.decode(coco.maskUtils.frPyObjects([ann[0]['segmentation']], 1024,1024))[:,:,0]
       
        a = cv2.resize(GT[0,:,:]  , (resize_par , resize_par),interpolation = cv2.INTER_CUBIC)
        b = cv2.resize(GT[1,:,:]  , (resize_par , resize_par),interpolation = cv2.INTER_CUBIC)

        GT = np.concatenate([a[None,:,:],b[None,:,:]],0)

        # seed = 1997
        # np.random.seed(seed) 
        # torch.manual_seed(seed)

        # if self.transform:
        #     im = self.transform(im)

        # np.random.seed(seed) 
        # torch.manual_seed(seed)

        # if self.transform:
        #     GT = self.transform(GT)

        # make sure to apply same tranform to both

        return torch.tensor(im).unsqueeze(0), torch.tensor(GT), len(annotations), img_id
         


    def __len__(self):
        return len(self.annotated)

    def __getitem__(self, index):
       

        return self.get_img_and_annotation(index)

class CNN3D(nn.Module):
    def __init__(self, input_channels, output_channels):
        super(CNN3D, self).__init__()

        self.input_channels = input_channels
        self.output_channels = output_channels

        # Encoder layers

        self.encoder_conv_00 = nn.Sequential(*[nn.Conv2d(in_channels=self.input_channels,out_channels=64,kernel_size=3,padding=1),nn.BatchNorm2d(64)])
        self.encoder_conv_01 = nn.Sequential(*[nn.Conv2d(in_channels=64,out_channels=64,kernel_size=3,padding=1),nn.BatchNorm2d(64)])

        self.encoder_conv_10 = nn.Sequential(*[nn.Conv2d(in_channels=64,out_channels=128,kernel_size=3,padding=1),nn.BatchNorm2d(128)])
        self.encoder_conv_11 = nn.Sequential(*[nn.Conv2d(in_channels=128,out_channels=128,kernel_size=3,padding=1),nn.BatchNorm2d(128)])

        self.encoder_conv_20 = nn.Sequential(*[nn.Conv2d(in_channels=128,out_channels=256,kernel_size=3,padding=1),nn.BatchNorm2d(256)])
        self.encoder_conv_21 = nn.Sequential(*[nn.Conv2d(in_channels=256,out_channels=256,kernel_size=3,padding=1),nn.BatchNorm2d(256)])
        self.encoder_conv_22 = nn.Sequential(*[nn.Conv2d(in_channels=256,out_channels=256,kernel_size=3,padding=1),nn.BatchNorm2d(256)])

        self.encoder_conv_30 = nn.Sequential(*[nn.Conv2d(in_channels=256,out_channels=512,kernel_size=3,padding=1),nn.BatchNorm2d(512)])
        self.encoder_conv_31 = nn.Sequential(*[nn.Conv2d(in_channels=512,out_channels=512,kernel_size=3,padding=1),nn.BatchNorm2d(512)])
        self.encoder_conv_32 = nn.Sequential(*[nn.Conv2d(in_channels=512,out_channels=512,kernel_size=3,padding=1),nn.BatchNorm2d(512)])

        self.encoder_conv_40 = nn.Sequential(*[nn.Conv2d(in_channels=512,out_channels=512,kernel_size=3,padding=1),nn.BatchNorm2d(512)])
        self.encoder_conv_41 = nn.Sequential(*[nn.Conv2d(in_channels=512,out_channels=512,kernel_size=3,padding=1),nn.BatchNorm2d(512)])
        self.encoder_conv_42 = nn.Sequential(*[nn.Conv2d(in_channels=512,out_channels=512,kernel_size=3,padding=1),nn.BatchNorm2d(512)])


        # Decoder layers

        self.decoder_convtr_42 = nn.Sequential(*[nn.ConvTranspose2d(in_channels=512,out_channels=512,kernel_size=3,padding=1),nn.BatchNorm2d(512)])
        self.decoder_convtr_41 = nn.Sequential(*[nn.ConvTranspose2d(in_channels=512,out_channels=512,kernel_size=3,padding=1),nn.BatchNorm2d(512)])
        self.decoder_convtr_40 = nn.Sequential(*[nn.ConvTranspose2d(in_channels=512,out_channels=512,kernel_size=3,padding=1),nn.BatchNorm2d(512)])

        self.decoder_convtr_32 = nn.Sequential(*[nn.ConvTranspose2d(in_channels=512,out_channels=512,kernel_size=3,padding=1),nn.BatchNorm2d(512)])
        self.decoder_convtr_31 = nn.Sequential(*[nn.ConvTranspose2d(in_channels=512,out_channels=512,kernel_size=3,padding=1),nn.BatchNorm2d(512)])
        self.decoder_convtr_30 = nn.Sequential(*[nn.ConvTranspose2d(in_channels=512,out_channels=256,kernel_size=3,padding=1),nn.BatchNorm2d(256)])

        self.decoder_convtr_22 = nn.Sequential(*[nn.ConvTranspose2d(in_channels=256,out_channels=256,kernel_size=3,padding=1),nn.BatchNorm2d(256)])
        self.decoder_convtr_21 = nn.Sequential(*[nn.ConvTranspose2d(in_channels=256,out_channels=256,kernel_size=3,padding=1),nn.BatchNorm2d(256)])
        self.decoder_convtr_20 = nn.Sequential(*[nn.ConvTranspose2d(in_channels=256,out_channels=128,kernel_size=3,padding=1),nn.BatchNorm2d(128)])

        self.decoder_convtr_11 = nn.Sequential(*[nn.ConvTranspose2d(in_channels=128,out_channels=128,kernel_size=3,padding=1),nn.BatchNorm2d(128)])
        self.decoder_convtr_10 = nn.Sequential(*[nn.ConvTranspose2d(in_channels=128,out_channels=64,kernel_size=3,padding=1),nn.BatchNorm2d(64)])

        self.decoder_convtr_01 = nn.Sequential(*[nn.ConvTranspose2d(in_channels=64,out_channels=64,kernel_size=3,padding=1),nn.BatchNorm2d(64)])
        self.decoder_convtr_00 = nn.Sequential(*[nn.ConvTranspose2d(in_channels=64,out_channels=self.output_channels,kernel_size=3,padding=1)])


        self.sigmoid = nn.Sigmoid()

    def forward(self, input_img):
        # Encoder Stage - 1
        dim_0 = input_img.size()
        x_00 = F.relu(self.encoder_conv_00(input_img))
        x_01 = F.relu(self.encoder_conv_01(x_00))
        x_0, indices_0 = F.max_pool2d(x_01, kernel_size=2, stride=2, return_indices=True)
        # Encoder Stage - 2
        dim_1 = x_0.size()
        x_10 = F.relu(self.encoder_conv_10(x_0))
        x_11 = F.relu(self.encoder_conv_11(x_10))
        x_1, indices_1 = F.max_pool2d(x_11, kernel_size=2, stride=2, return_indices=True)
        # Encoder Stage - 3
        dim_2 = x_1.size()
        x_20 = F.relu(self.encoder_conv_20(x_1))
        x_21 = F.relu(self.encoder_conv_21(x_20))
        x_22 = F.relu(self.encoder_conv_22(x_21))
        x_2, indices_2 = F.max_pool2d(x_22, kernel_size=2, stride=2, return_indices=True)
        # Encoder Stage - 4
        dim_3 = x_2.size()
        x_30 = F.relu(self.encoder_conv_30(x_2))
        x_31 = F.relu(self.encoder_conv_31(x_30))
        x_32 = F.relu(self.encoder_conv_32(x_31))
        x_3, indices_3 = F.max_pool2d(x_32, kernel_size=2, stride=2, return_indices=True)
        
        # Encoder Stage - 5
        dim_4 = x_3.size()
        x_40 = F.relu(self.encoder_conv_40(x_3))
        x_41 = F.relu(self.encoder_conv_41(x_40))
        x_42 = F.relu(self.encoder_conv_42(x_41))
        x_4, indices_4 = F.max_pool2d(x_42, kernel_size=2, stride=2, return_indices=True)
        
        # Decoder


        # Decoder Stage - 5
        x_4d = F.max_unpool2d(x_4, indices_4, kernel_size=2, stride=2, output_size=dim_4)
        x_42d = F.relu(self.decoder_convtr_42(x_4d))
        x_41d = F.relu(self.decoder_convtr_41(x_42d))
        x_40d = F.relu(self.decoder_convtr_40(x_41d))

        x_40d = x_40d + x_3

        # Decoder Stage - 4
        x_3d = F.max_unpool2d(x_40d, indices_3, kernel_size=2, stride=2, output_size=dim_3)
        x_32d = F.relu(self.decoder_convtr_32(x_3d))
        x_31d = F.relu(self.decoder_convtr_31(x_32d))
        x_30d = F.relu(self.decoder_convtr_30(x_31d))

        x_30d = x_30d + x_2

        # Decoder Stage - 3
        x_2d = F.max_unpool2d(x_30d, indices_2, kernel_size=2, stride=2, output_size=dim_2)
        x_22d = F.relu(self.decoder_convtr_22(x_2d))
        x_21d = F.relu(self.decoder_convtr_21(x_22d))
        x_20d = F.relu(self.decoder_convtr_20(x_21d))

        x_20d = x_20d + x_1

        # Decoder Stage - 2
        x_1d = F.max_unpool2d(x_20d, indices_1, kernel_size=2, stride=2, output_size=dim_1)
        x_11d = F.relu(self.decoder_convtr_11(x_1d))
        x_10d = F.relu(self.decoder_convtr_10(x_11d))


        x_10d = x_10d + x_0

        # Decoder Stage - 1
        x_0d = F.max_unpool2d(x_10d, indices_0, kernel_size=2, stride=2, output_size=dim_0)

        x_01d = F.relu(self.decoder_convtr_01(x_0d))
        x_00d = self.decoder_convtr_00(x_01d)


        return self.sigmoid(x_00d)
    



if __name__ == "__main__":
    device = torch.device("cpu")
    if(torch.backends.mps.is_available()):
        device = torch.device("mps")
    elif(torch.cuda.is_available()):
        device = torch.device("cuda")



    model = CNN3D(4,2).to(device)

    # composed = v2.Compose([v2.RandomHorizontalFlip(p=0.5), v2.RandomRotation((0, 360)), v2.RandomVerticalFlip(p=0.5)])
    # same number cmes train test
    dataset = FrondandDiff()
            
    # # split the data set into train and validation
    train_indices = dataset.train_data_idx
    test_indices =  dataset.test_data_idx

    # why are they not in the right order?

    train_dataset = torch.utils.data.Subset(dataset, train_indices)
    test_dataset = torch.utils.data.Subset(dataset, test_indices)

    train_data_loader = torch.utils.data.DataLoader(
                                                train_dataset,
                                                batch_size=1,
                                                shuffle=False,
                                                num_workers=1,
                                                pin_memory=False
                                            )
    
    g_optimizer = optim.Adam(model.parameters(),1e-4)
    # BCEwithlogitloss
    pixel_looser= nn.BCELoss()

    for i in range(0,200):
        predictions = torch.tensor(np.zeros((1, 2, 256, 256)))
        im_prev = torch.tensor(np.zeros((1, 1, 256, 256)))
        ind_prev = -0.5

        for p, data in enumerate(train_data_loader, 0):
            
            start = time.time()
            g_optimizer.zero_grad()

            input_data = torch.cat([data[0].float().to(device), predictions.float().to(device), im_prev.float().to(device)], dim=1)

            pred = model(input_data)

            if (data[2] > 0) & ((data[3] - ind_prev) == 1.):
                predictions = pred.detach()
                im_prev = data[0].detach()

            elif (data[2] > 0) & (ind_prev == -0.5):
                predictions = pred.detach()
                im_prev = data[0].detach()

            else:
                predictions = torch.tensor(np.zeros(np.shape(pred)))
                im_prev = torch.tensor(np.zeros(np.shape(data[0])))

            ind_prev = float(data[3].detach())

            loss = pixel_looser(pred, data[1].float().to(device))
            loss.backward()
            g_optimizer.step()
            print(loss,time.time()-start)

            fig,ax = plt.subplots(2,3)
            ax[0][0].imshow(data[0][0][0].detach().cpu().numpy())
            ax[0][1].imshow(data[1][0][0].detach().cpu().numpy())
            ax[0][2].imshow(data[1][0][1].detach().cpu().numpy())
            ax[1][0].imshow(data[0][0][0].detach().cpu().numpy())
            ax[1][1].imshow(pred[0][0].detach().cpu().numpy())
            ax[1][2].imshow(pred[0][1].detach().cpu().numpy())
            plt.savefig('test_'+str(i)+'.png')
            # load model / save model weights, autooptimizer (g_optimizer)

            
    torch.save(model.state_dict(), 'model.pth')
