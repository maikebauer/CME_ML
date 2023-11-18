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
import evaluation

class FrondandDiff(Dataset):
    def __init__(self, transform=None):

        self.transform = transform

        self.coco_obj = coco.COCO("instances_default.json")
        

        self.img_ids = self.coco_obj.getImgIds()[:3844]###until last image annotated

        self.annotated = []
        self.events = []

        event_list = []
        temp_list = []

        a_id_prev = 0

        for a in range(0, len(self.img_ids)):
            anns = self.coco_obj.getAnnIds(imgIds=[self.img_ids[a]])
            

            if(len(anns)>0):

                self.annotated.append(a)

                if (self.img_ids[a] - a_id_prev == 1) or (a_id_prev == 0):

                    temp_list.append(a)

                elif (self.img_ids[a] - a_id_prev) > 1:

                    if len(temp_list) > 0:
                        event_list.append(temp_list)

                    temp_list = []
         
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

        self.train_data_idx = sorted(train_data)
        self.test_data_idx = sorted(test_data)
        

    def get_img_and_annotation(self,idx):

        width_par = 1024
        height_par = width_par

        img_id   = self.img_ids[idx]
        img_info = self.coco_obj.loadImgs([img_id])[0]
        img_file_name = img_info["file_name"]

        if torch.cuda.is_available():
            path = "/gpfs/data/fs72241/maibauer/differences/"

        else:
            path = "/Volumes/SSD/differences/"

        # Use URL to load image.
        im = np.asarray(Image.open(path+img_file_name).convert("L"))/255.0

        if width_par != 1024:
            im = cv2.resize(im  , (width_par , height_par),interpolation = cv2.INTER_CUBIC)

        GT = np.zeros((2,1024,1024))
        annotations = self.coco_obj.getAnnIds(imgIds=img_id)

        if(len(annotations)>0):
            for a in annotations:
                ann = self.coco_obj.loadAnns(a)
                GT[int(ann[0]["attributes"]["id"]),:,:]=coco.maskUtils.decode(coco.maskUtils.frPyObjects([ann[0]['segmentation']], 1024, 1024))[:,:,0]
        
        if width_par != 1024:
            a = cv2.resize(GT[0,:,:]  , (width_par , height_par),interpolation = cv2.INTER_CUBIC)
            b = cv2.resize(GT[1,:,:]  , (width_par , height_par),interpolation = cv2.INTER_CUBIC)

        else:
            a = GT[0,:,:]
            b = GT[1,:,:]

        GT = np.concatenate([a[None,:,:],b[None,:,:]],0)
        # GT = np.sum(GT, axis=0)

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

        return torch.tensor(im).unsqueeze(0), torch.tensor(GT)
         


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

        return x_00d
        #rn -> two classes (mask0, mask1) -> softmax to ensure probabilities add up to 1
        #or -> one class (add mask0, mask1) -> sigmoid to ensure probabilities are between 0 and 1
        #return self.sigmoid(x_00d)
    

def train():

    device = torch.device("cpu")

    if(torch.backends.mps.is_available()):
        device = torch.device("mps")

    elif(torch.cuda.is_available()):
        device = torch.device("cuda")

    model = CNN3D(1,2).to(device)

    # composed = v2.Compose([v2.RandomHorizontalFlip(p=0.5), v2.RandomRotation((0, 360)), v2.RandomVerticalFlip(p=0.5)])
    dataset = FrondandDiff()
            
    indices = dataset.train_data_idx

    dataset_sub = torch.utils.data.Subset(dataset, indices)

    batch_size = 4
    num_workers = 2

    data_loader = torch.utils.data.DataLoader(
                                                dataset_sub,
                                                batch_size=batch_size,
                                                shuffle=False,
                                                num_workers=num_workers,
                                                pin_memory=False
                                            )
    
    g_optimizer = optim.Adam(model.parameters(),1e-4)

    num_iter = 200

    for epoch in range(0, num_iter):
        for p, data in enumerate(data_loader, 0):
            
            start = time.time()
            g_optimizer.zero_grad()

            input_data = data[0].float().to(device)
            mask_data = data[1].float().to(device)

            pred = model(input_data)

            weights = np.zeros((np.shape(mask_data)[1]))

            for i, channel in enumerate(range(len(weights))):
                if mask_data[:,channel,:,:].sum() == 0:
                    weights[i] = 1
                else:
                    weights[i] = (mask_data[:,channel,:,:]==0).sum()/mask_data[:,channel,:,:].sum()
            
            weights = weights[:,None,None]

            weights = torch.tensor(weights).to(device, dtype=torch.float32)
            pixel_looser= nn.BCEWithLogitsLoss(pos_weight=weights)

            loss = pixel_looser(pred, mask_data)

            loss.backward()
            g_optimizer.step()
            
            #print(loss,time.time()-start)

            fig,ax = plt.subplots(2,3)
            ax[0][0].imshow(data[0][0][0].detach().cpu().numpy()) #image
            ax[0][1].imshow(data[1][0][0].detach().cpu().numpy()) #mask0
            ax[0][2].imshow(data[1][0][1].detach().cpu().numpy()) #mask1
            ax[1][0].imshow(data[0][0][0].detach().cpu().numpy()) 
            ax[1][1].imshow(pred[0][0].detach().cpu().numpy()) #mask0
            ax[1][2].imshow(pred[0][1].detach().cpu().numpy()) #mask1

            plt.savefig('test_'+str(epoch)+'.png')
            plt.close()

        torch.save(model.state_dict(), 'model_test.pth')

@torch.no_grad()
def test():

    device = torch.device("cpu")

    if(torch.backends.mps.is_available()):
        device = torch.device("mps")

    elif(torch.cuda.is_available()):
        device = torch.device("cuda")

    model = CNN3D(1,2).to(device)
    model.load_state_dict(torch.load('model_test.pth', map_location=device))
    model.eval()

    # composed = v2.Compose([v2.RandomHorizontalFlip(p=0.5), v2.RandomRotation((0, 360)), v2.RandomVerticalFlip(p=0.5)])
    dataset = FrondandDiff()
            
    indices = dataset.test_data_idx
    dataset_sub = torch.utils.data.Subset(dataset, indices)

    batch_size = 1
    num_workers = 1

    data_loader = torch.utils.data.DataLoader(
                                                dataset_sub,
                                                batch_size=batch_size,
                                                shuffle=False,
                                                num_workers=num_workers,
                                                pin_memory=False
                                            )
    save_metrics = []

    for p, data in enumerate(data_loader, 0):
        
        input_data = data[0].float().to(device)

        pred = model(input_data)
        metrics = evaluation.evaluate(pred[0].cpu().detach(),data[1][0].numpy(),data[0][0][0].cpu().detach())

        save_metrics.append(metrics)

    print('Saving metrics...')
    np.save('metrics_test.npy', save_metrics)

if __name__ == "__main__":
    train()
    # test()
