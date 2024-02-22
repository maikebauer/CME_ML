import torch
from torch.utils.data import Dataset
from torchvision import transforms
import glob
import matplotlib.pyplot as plt 
import numpy as np
from PIL import Image
import cv2
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')
import pickle
# from astropy.wcs import WCS
from scipy.ndimage import shift


class Beacon2ScienceDataset(Dataset):
    def __init__(self, l_resolution=128, r_resolution=256,path="visus_test",training=True):
        self.l_res = l_resolution
        self.r_res = r_resolution
        self.paths_lr = sorted(glob.glob(path+"/beacon/*"))
        self.paths_hr = sorted(glob.glob(path+"/science/*"))


        if(training):
            self.paths_lr = self.paths_lr[:int(len(self.paths_lr)*0.7)]
            self.paths_hr = self.paths_hr[:int(len(self.paths_hr)*0.7)]
        
      

    def __len__(self):
        return len(self.paths_lr)

    def __getitem__(self, index):
        img_low_res = np.asarray(Image.open(self.paths_lr[index]))/255.0
        img_low_res = cv2.resize(img_low_res  , (self.l_res , self.l_res),interpolation = cv2.INTER_CUBIC)
        
        img_high_res = np.asarray(Image.open(self.paths_hr[index]))/255.0
        img_high_res = cv2.resize(img_high_res  , (self.r_res , self.r_res),interpolation = cv2.INTER_CUBIC)


        return {"LR":torch.tensor(img_low_res).unsqueeze(0).float(),"HR":torch.tensor(img_high_res).unsqueeze(0).float()}
     


class CombinedDataloader(Dataset):
    def __init__(self, l_resolution=128, r_resolution=256,path="/Volumes/PortableSSD/images3",training=True,validation=False):
        self.l_res = l_resolution
        self.r_res = r_resolution
        # print(path)
        # print(path+"/*")
        # print("/gpfs/data/fs72241/lelouedecj/images3"+"/*")
        self.paths = sorted(glob.glob(path+"/*"))
        # dataset = b2s_dataset.CombinedDataloader(256,1024,"/gpfs/data/fs72241/lelouedecj/images3",True,False)
        # /gpfs/data/fs72241/lelouedecj/images3


        print(len(self.paths))

        self.shifts = np.load("shifts2.npy")


        # if(training):
        #     self.paths = self.paths[:int(self.shifts.shape[0]*0.7)]
        #     self.shifts = self.shifts[:int(self.shifts.shape[0]*0.7)]
        # elif(validation):
        #     self.paths = self.paths[int(self.shifts.shape[0]*0.7):int(self.shifts.shape[0]*0.9)]
        #     self.shifts = self.shifts[int(self.shifts.shape[0]*0.7):int(self.shifts.shape[0]*0.9)]
        # else:
        #     self.paths = self.paths[int(self.shifts.shape[0]*0.9):]
        #     self.shifts = self.shifts[int(self.shifts.shape[0]*0.9):]




        if(training):
            self.paths = self.paths[-400:-200]
            self.shifts = self.shifts[-400:-200]
        elif(validation):
            self.paths = self.paths[-200:-100]
            self.shifts = self.shifts[-200:-100]
        else:
            self.paths = self.paths[-100:]
            self.shifts = self.shifts[-100:]

        print(len(self.paths),self.shifts.shape)

        # self.clahe = cv2.createCLAHE(clipLimit=2,tileGridSize=(8,8))
        
      

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        img = Image.open(self.paths[index])
        wholeimg = np.asarray(img)
        img.close()

        b1  =  wholeimg[:256,:256]
        b2  =  wholeimg[256:512,:256]
        s25 =  wholeimg[1024:,1024:2048]
        s75 =  wholeimg[1024:,2048:]
        s1  =  wholeimg[:1024,1024:2048]
        s2  =  wholeimg[:1024,2048:]


        clahe = cv2.createCLAHE(clipLimit=3,tileGridSize=(10,10))
        s1  = clahe.apply(s1)/255.0
        b1  = clahe.apply(b1)/255.0
        s2  = clahe.apply(s2)/255.0
        b2  = clahe.apply(b2)/255.0
        s25 =  clahe.apply(s25)/255.0
        s75 =  clahe.apply(s75)/255.0



        # headers_path = '/gpfs/data/fs72241/lelouedecj/final_data/'+self.paths[index].split("/")[-1][:-4]+".pkl"
        # with open(headers_path, 'rb') as fi:
        #     element = pickle.load(fi)
        #     fi.close()
        # print(self.paths[index],":",len(element["dates"]))
        shifts = self.shifts[index]
        shifts = [shifts[1],shifts[0]]
        diff1,tr1 = self.difference(s2,s1 ,shifts)
        diff2,tr2 = self.difference(s1,s2 ,-1*np.array(shifts))

        tr1 = np.array([tr1[1],tr1[0]])

        # diff2,tr2 = self.difference(s75,s25,shifts[1])
        # diff3,tr3 = self.difference(s2,s75 ,shifts[2])


        diff1_b,tr1_b = self.difference(b2,b1 ,np.array(shifts)/4)
        diff2_b,tr2_b = self.difference(b1,b2 ,-1*np.array(shifts)/4)





        b1  =  cv2.resize(b1, (self.l_res , self.l_res),interpolation = cv2.INTER_CUBIC) #beacon1
        b1  = (b1-b1.min())/(b1.max()-b1.min()) 
        b2  =  cv2.resize(b2, (self.l_res , self.l_res),interpolation = cv2.INTER_CUBIC) #beacon2
        b2  = (b2-b2.min())/(b2.max()-b2.min()) 
        s25 =  cv2.resize(s25, (self.r_res , self.r_res),interpolation = cv2.INTER_CUBIC)
        s25  = (s25-s25.min())/(s25.max()-s25.min()) 
        s75 =  cv2.resize(s75, (self.r_res , self.r_res),interpolation = cv2.INTER_CUBIC)
        s75  = (s75-s75.min())/(s75.max()-s75.min()) 
        s1  =  cv2.resize(s1, (self.r_res , self.r_res),interpolation = cv2.INTER_CUBIC)
        s1  = (s1-s1.min())/(s1.max()-s1.min()) 
        s2  =  cv2.resize(s2, (self.r_res , self.r_res),interpolation = cv2.INTER_CUBIC)
        s1  = (s1-s1.min())/(s1.max()-s1.min()) 


        diff1 = cv2.resize(diff1, (self.r_res , self.r_res),interpolation = cv2.INTER_CUBIC)
        diff1  = (diff1-diff1.min())/(diff1.max()-diff1.min())
        diff2 = cv2.resize(diff2, (self.r_res , self.r_res),interpolation = cv2.INTER_CUBIC)
        diff2  = (diff2-diff2.min())/(diff2.max()-diff2.min())


        return {"LR1":torch.tensor(b1).unsqueeze(0).float(), #beacon1
                "LR2":torch.tensor(b2).unsqueeze(0).float(), #beacon2
                "HR1":torch.tensor(s1).float(), #science1
                "HR2":torch.tensor(s2).float(), #science2
                "M1":torch.tensor(s25).float(), #science.25
                "M2":torch.tensor(s75).float(), #science.75
                "diff1":torch.tensor(diff1), #diff science1 science2
                "diff2":torch.tensor(diff2), #diff science2 science1
                "tr1":tr1, #shifted science1 science2 with shift header
                "diff1_b":torch.tensor(diff1_b), #diff beacon1 beacon2
                "diff2_b":torch.tensor(diff2_b), #diff beacon2 beacon1
               
                }
    
        # return {"LR1":torch.tensor(b1).unsqueeze(0).float(),
        #     "LR2":torch.tensor(b2).unsqueeze(0).float(),
        #     "HR1":torch.tensor(s1).float(),
        #     "HR2":torch.tensor(s2).float(),
        #     "M1":torch.tensor(s25).float(),
        #     "M2":torch.tensor(s75).float(),
        #     "diff1":torch.tensor(diff1),
        #     # "diff2":torch.tensor(diff2),
        #     # "diff3":torch.tensor(diff3),
        #     "tr1":tr1,
        #     # "tr2":tr2,
        #     # "tr3":tr3
        #     }


    def difference(self,img,img_prev,shift_arr):

        # center      = header['crpix1']-1, header['crpix2']-1
        # wcs = WCS(header,key='A')
        # center_prev = wcs.all_world2pix(header_prev["crval1a"],header_prev["crval2a"], 0)
        # shift_arr = [center_prev[1]-center[1],(center_prev[0]-center[0])]
        difference   = np.float32(img-shift(img_prev,shift_arr, order=2,mode='nearest',prefilter=False))
        difference = (difference- difference.min())/(difference.max()-difference.min())
        
        # difference = (difference*255.0).astype(np.uint8)
        # clahe = cv2.createCLAHE(clipLimit=10,tileGridSize=(8,8))
        # difference = clahe.apply(difference)

        return difference,shift_arr


class CombinedDataloader3(Dataset):
    def __init__(self, l_resolution=128, r_resolution=256,path="../finals_test",training=True,validation=False):
        self.l_res = l_resolution
        self.r_res = r_resolution
       
        self.paths = sorted(glob.glob(path+"/*"))
        # dataset = b2s_dataset.CombinedDataloader(256,1024,"/gpfs/data/fs72241/lelouedecj/images3",True,False)
        # /gpfs/data/fs72241/lelouedecj/images3


        print(len(self.paths))

        self.shifts = np.load("shifts2.npy")

        if(training):
            self.paths = self.paths[-3500:-600]
            self.shifts = self.shifts[-3500:-600]
        elif(validation):
            self.paths = self.paths[-600:-400]
            self.shifts = self.shifts[-600:-400]
        else:
            self.paths = self.paths[-400:]
            self.shifts = self.shifts[-400:]

        print(len(self.paths),self.shifts.shape)

        # self.clahe = cv2.createCLAHE(clipLimit=2,tileGridSize=(8,8))
        
      

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):

        with open(self.paths[index], 'rb') as fi:
            data = pickle.load(fi)
        fi.close()



        clahe_b = cv2.createCLAHE(clipLimit=10,tileGridSize=(4,4))
        clahe_s = cv2.createCLAHE(clipLimit=10,tileGridSize=(10,10))
        s1  = clahe_s.apply(data['s1'])/255.0
        b1  = clahe_b.apply(data['b1'])/255.0
        s2  = clahe_s.apply(data['s2'])/255.0
        b2  = clahe_b.apply(data['b2'])/255.0
        s25 = clahe_s.apply(data['s125'])/255.0
        s75 = clahe_s.apply(data['s175'])/255.0


        # s1  = data['s1']/255.0
        # b1  = data['b1']/255.0
        # s2  = data['s2']/255.0
        # b2  = data['b2']/255.0
        # s25 = data['s125']/255.0
        # s75 = data['s175']/255.0


        


        # headers_path = '../../final_data/'+self.paths[index].split("/")[-1][:-4]+".pkl"
        # with open(headers_path, 'rb') as fi:
        #     element = pickle.load(fi)
        #     fi.close()

        # element = element["dates"]

        # print(self.paths[index],":",len(element["dates"]))
        shifts = self.shifts[index]
        shifts = [shifts[1],shifts[0]]
        diff1,tr1 = self.difference(s2,s1 ,shifts)#,element[5],element[1])
        diff2,tr2 = self.difference(s1,s2 ,-1*np.array(shifts))
        

        tr1 = np.array([tr1[1],tr1[0]])

      


        diff1_b,tr1_b = self.difference(b2,b1 ,np.array(shifts))
        diff2_b,tr2_b = self.difference(b1,b2,(-1*np.array(shifts))/4)



        # b1  =  cv2.resize(b1, (self.l_res , self.l_res),interpolation = cv2.INTER_CUBIC)
        # b1  = (b1-b1.min())/(b1.max()-b1.min()) 

        # b2  =  cv2.resize(b2, (self.l_res , self.l_res),interpolation = cv2.INTER_CUBIC)
        # b2  = (b2-b2.min())/(b2.max()-b2.min()) 

        s25 =  cv2.resize(s25, (self.r_res , self.r_res),interpolation = cv2.INTER_CUBIC)
        # s25  = (s25-s25.min())/(s25.max()-s25.min()) 
        s75 =  cv2.resize(s75, (self.r_res , self.r_res),interpolation = cv2.INTER_CUBIC)
        # s75  = (s75-s75.min())/(s75.max()-s75.min()) 
        s1  =  cv2.resize(s1, (self.r_res , self.r_res),interpolation = cv2.INTER_CUBIC)
        # s1  = (s1-s1.min())/(s1.max()-s1.min()) 
        s2  =  cv2.resize(s2, (self.r_res , self.r_res),interpolation = cv2.INTER_CUBIC)
        # s1  = (s1-s1.min())/(s1.max()-s1.min()) 


        diff1 = cv2.resize(diff1, (self.r_res , self.r_res),interpolation = cv2.INTER_CUBIC)
        # diff1  = (diff1-diff1.min())/(diff1.max()-diff1.min())
        diff2 = cv2.resize(diff2, (self.r_res , self.r_res),interpolation = cv2.INTER_CUBIC)
        # diff2  = (diff2-diff2.min())/(diff2.max()-diff2.min())

        


        return {"LR1":torch.tensor(b1).unsqueeze(0).float(),
                "LR2":torch.tensor(b2).unsqueeze(0).float(),
                "HR1":torch.tensor(s1).unsqueeze(0).float(),
                "HR2":torch.tensor(s2).unsqueeze(0).float(),
                "M1":torch.tensor(s25).unsqueeze(0).float(),
                "M2":torch.tensor(s75).unsqueeze(0).float(),
                "diff1":torch.tensor(diff1).unsqueeze(0).float(),
                "diff2":torch.tensor(diff2).unsqueeze(0).float(),
                "tr1":tr1,
                "diff1_b":torch.tensor(diff1_b).unsqueeze(0).float(),
                "diff2_b":torch.tensor(diff2_b).unsqueeze(0).float(),
               
                }
    


    def difference(self,img,img_prev,shift_arr):#header,header_prev):

        # center      = header['crpix1']-1, header['crpix2']-1
        # print(header_prev["crval1a"],header_prev["crval2a"],header["DATE-OBS"],header["DATE-END"],center)
        # wcs = WCS(header,key='A')
        # center_prev = wcs.all_world2pix(header_prev["crval1a"],header_prev["crval2a"], 0)
        # shift_arr = [center_prev[1]-center[1],(center_prev[0]-center[0])]
        difference   = np.float32(img-shift(img_prev,shift_arr, order=2,mode='nearest',prefilter=False))
        # difference = (differe nce- difference.min())/(difference.max()-difference.min())
        
        # difference = (difference*255.0).astype(np.uint8)
        # clahe = cv2.createCLAHE(clipLimit=10,tileGridSize=(8,8))
        # difference = clahe.apply(difference)

        return difference,shift_arr    


      

     



if __name__ == "__main__":
    dataset = CombinedDataloader(256,512,"../images3",True)

    # shifts  = []
    for i in range(0,dataset.__len__()):
        fig,ax= plt.subplots(1,3)
        ax[0].imshow(dataset.__getitem__(i)["diff1"].cpu().numpy(),cmap='gray')
        ax[0].axis("off")
        ax[0].set_title("shift im_prev")
        ax[1].imshow(dataset.__getitem__(i)["diff2"].cpu().numpy(),cmap='gray')
        ax[1].axis("off")
        ax[1].set_title("shift im")
        ax[2].imshow(dataset.__getitem__(i)["diff1"].cpu().numpy()-dataset.__getitem__(i)["diff2"].cpu().numpy(),cmap='gray')
        ax[2].axis("off")
        ax[2].set_title("diff differences")
        plt.show()
    