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


class FrondandDiff(Dataset):
    def __init__(self, training=True):

        self.coco_obj = coco.COCO("instances_default.json")
        

        self.img_ids = self.coco_obj.getImgIds()[:3844]###until last image annotated

        

        # self.ann_ids = self.coco_obj.getAnnIds(iscrowd=None)
        self.annotated = []
        for a in range(0,len(self.img_ids)):
            anns = self.coco_obj.getAnnIds(imgIds=[self.img_ids[a]])
            if(len(anns)>0):
                 self.annotated.append(a)

     
        self.not_annotated = np.setdiff1d(np.arange(0,len(self.img_ids)),np.array(self.annotated))

        self.not_annotated = self.not_annotated[np.random.choice(len(self.not_annotated), len(self.annotated))]

        self.annotated = list(self.annotated) + list(self.not_annotated)
    


        # if(training):
        #     self.annot_paths = self.annot_paths[:int(len(self.annot_paths)*0.7)]
        
    

    def get_img_and_annotation(self,idx):

        resize_par = 256

        # Pick one image.
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


        return torch.tensor(im).unsqueeze(0),torch.tensor(GT)
         


    def __len__(self):
        return len(self.annotated)

    def __getitem__(self, index):
       

        return self.get_img_and_annotation(index)
     


class UNet(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        
        # Encoder
        # In the encoder, convolutional layers with the Conv2d function are used to extract features from the input image. 
        # Each block in the encoder consists of two convolutional layers followed by a max-pooling layer, with the exception of the last block which does not include a max-pooling layer.
        # -------
        # input: 572x572x3

        self.in_ch = in_ch
        self.out_ch = out_ch
         
        self.e11 = nn.Sequential(*[nn.Conv2d(in_channels=self.in_ch,out_channels=64,kernel_size=3,padding=1),nn.BatchNorm2d(64)])
        self.e12 = nn.Sequential(*[nn.Conv2d(in_channels=64,out_channels=64,kernel_size=3,padding=1),nn.BatchNorm2d(64)])
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2) # output: 284x284x64

        # input: 284x284x64
        self.e21 = nn.Sequential(*[nn.Conv2d(in_channels=64,out_channels=128,kernel_size=3,padding=1),nn.BatchNorm2d(128)])
        self.e22 = nn.Sequential(*[nn.Conv2d(in_channels=128,out_channels=128,kernel_size=3,padding=1),nn.BatchNorm2d(128)])
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2) # output: 140x140x128

        # input: 140x140x128
        self.e31 = nn.Sequential(*[nn.Conv2d(in_channels=128,out_channels=256,kernel_size=3,padding=1),nn.BatchNorm2d(256)])
        self.e32 = nn.Sequential(*[nn.Conv2d(in_channels=256,out_channels=256,kernel_size=3,padding=1),nn.BatchNorm2d(256)])
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2) # output: 68x68x256

        # input: 68x68x256
        self.e41 = nn.Sequential(*[nn.Conv2d(in_channels=256,out_channels=512,kernel_size=3,padding=1),nn.BatchNorm2d(512)])
        self.e42 = nn.Sequential(*[nn.Conv2d(in_channels=512,out_channels=512,kernel_size=3,padding=1),nn.BatchNorm2d(512)])
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2) # output: 32x32x512

        # input: 32x32x512
        self.e51 =  nn.Sequential(*[nn.Conv2d(in_channels=512,out_channels=1024,kernel_size=3,padding=1),nn.BatchNorm2d(1024)])
        self.e52 = nn.Sequential(*[nn.Conv2d(in_channels=1024,out_channels=1024,kernel_size=3,padding=1),nn.BatchNorm2d(1024)])


        # Decoder
        self.upconv1 = nn.Sequential(*[nn.ConvTranspose2d(in_channels=1024,out_channels=512, kernel_size=2, stride=2),nn.BatchNorm2d(512)])
        self.d11 = nn.Sequential(*[nn.Conv2d(in_channels=1024,out_channels=512, kernel_size=3, padding=1),nn.BatchNorm2d(512)])
        self.d12 = nn.Sequential(*[nn.Conv2d(in_channels=512,out_channels=512, kernel_size=3, padding=1),nn.BatchNorm2d(512)])

        self.upconv2 = nn.Sequential(*[nn.ConvTranspose2d(in_channels=512,out_channels=256, kernel_size=2, stride=2),nn.BatchNorm2d(256)])
        self.d21 = nn.Sequential(*[nn.Conv2d(in_channels=512,out_channels=256, kernel_size=3, padding=1),nn.BatchNorm2d(256)])
        self.d22 = nn.Sequential(*[nn.Conv2d(in_channels=256,out_channels=256, kernel_size=3, padding=1),nn.BatchNorm2d(256)])

        self.upconv3 = nn.Sequential(*[nn.ConvTranspose2d(in_channels=256,out_channels=128, kernel_size=2, stride=2),nn.BatchNorm2d(128)])
        self.d31 = nn.Sequential(*[nn.Conv2d(in_channels=256,out_channels=128, kernel_size=3, padding=1),nn.BatchNorm2d(128)])
        self.d32 = nn.Sequential(*[nn.Conv2d(in_channels=128,out_channels=128, kernel_size=3, padding=1),nn.BatchNorm2d(128)])

        self.upconv4 = nn.Sequential(*[nn.ConvTranspose2d(in_channels=128,out_channels=64, kernel_size=2, stride=2),nn.BatchNorm2d(64)])
        self.d41 = nn.Sequential(*[nn.Conv2d(in_channels=128,out_channels=64, kernel_size=3, padding=1),nn.BatchNorm2d(64)])
        self.d42 = nn.Sequential(*[nn.Conv2d(in_channels=64,out_channels=64, kernel_size=3, padding=1),nn.BatchNorm2d(64)])

        # Output layer
        self.outconv = nn.Conv2d(in_channels=64,out_channels=self.out_ch, kernel_size=1)
        # self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Encoder
        xe11 =  F.relu(self.e11(x))
        xe12 =  F.relu(self.e12(xe11))
        xp1 = self.pool1(xe12)

        xe21 =  F.relu(self.e21(xp1))
        xe22 =  F.relu(self.e22(xe21))
        xp2 = self.pool2(xe22)

        xe31 =  F.relu(self.e31(xp2))
        xe32 =  F.relu(self.e32(xe31))
        xp3 = self.pool3(xe32)

        xe41 =  F.relu(self.e41(xp3))
        xe42 =  F.relu(self.e42(xe41))
        xp4 = self.pool4(xe42)

        xe51 =  F.relu(self.e51(xp4))
        xe52 =  F.relu(self.e52(xe51))
        
        # Decoder
        xu1 = self.upconv1(xe52)
        xu11 = torch.cat([xu1, xe42], dim=1)
        xd11 =  F.relu(self.d11(xu11))
        xd12 =  F.relu(self.d12(xd11))

        xu2 = self.upconv2(xd12)
        xu22 = torch.cat([xu2, xe32], dim=1)
        xd21 =  F.relu(self.d21(xu22))
        xd22 =  F.relu(self.d22(xd21))

        xu3 = self.upconv3(xd22)
        xu33 = torch.cat([xu3, xe22], dim=1)
        xd31 =  F.relu(self.d31(xu33))
        xd32 =  F.relu(self.d32(xd31))

        xu4 = self.upconv4(xd32)
        xu44 = torch.cat([xu4, xe12], dim=1)
        xd41 =  F.relu(self.d41(xu44))
        xd42 =  F.relu(self.d42(xd41))

        # Output layer
        out = self.outconv(xd42)
        return out


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



    # model = CNN3D(1,2).to(device)
    model = UNet(1, 2).to(device)

    dataset = FrondandDiff(True)
    # for i in range(0,dataset.__len__()):
    #     im,mask = dataset.__getitem__(i)
    #     fig,ax = plt.subplots(1,3)
    #     ax[0].imshow(im)
    #     ax[1].imshow(mask[0])
    #     ax[2].imshow(mask[1])
    #     plt.show()
    # exit()
    dataloader = torch.utils.data.DataLoader(
                                                dataset,
                                                batch_size=1,
                                                shuffle=False,
                                                num_workers=1,
                                                pin_memory=False
                                            )
    

    g_optimizer = optim.Adam(model.parameters(),1e-4)
    pixel_looser= nn.BCELoss()

    for i in range(0,50):
        for data in dataloader:

            start = time.time()
            g_optimizer.zero_grad()

            pred = model(data[0].float().to(device))


            loss = pixel_looser(pred,data[1].float().to(device))
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

            
            
