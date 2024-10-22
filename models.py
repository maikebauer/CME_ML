
from typing import Tuple, Union
import torch
import torch.nn as nn
from monai.networks.blocks import UnetrBasicBlock, UnetrPrUpBlock, UnetrUpBlock
from monai.networks.blocks.dynunet_block import UnetOutBlock
from monai.networks.nets import ViT
import torch.nn.functional as F
from utils import calc_kernel
from modules import (
    ResidualConv,
    ASPP,
    AttentionBlock,
    Upsample_,
    Squeeze_Excite_Block,
)


kernel_initializer = 'he_uniform'
interpolation = "nearest"

class UNETR_16(nn.Module):
    """
    UNETR based on: "Hatamizadeh et al.,
    UNETR: Transformers for 3D Medical Image Segmentation <https://arxiv.org/abs/2103.10504>"
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        img_size: Tuple[int, int, int],
        feature_size: int = 16,
        hidden_size: int = 768,
        mlp_dim: int = 3072,
        num_heads: int = 12,
        pos_embed: str = "perceptron",
        norm_name: Union[Tuple, str] = "instance",
        conv_block: bool = False,
        res_block: bool = True,
        dropout_rate: float = 0.0,
    ) -> None:
        """
        Args:
            in_channels: dimension of input channels.
            out_channels: dimension of output channels.
            img_size: dimension of input image.
            feature_size: dimension of network feature size.
            hidden_size: dimension of hidden layer.
            mlp_dim: dimension of feedforward layer.
            num_heads: number of attention heads.
            pos_embed: position embedding layer type.
            norm_name: feature normalization type and arguments.
            conv_block: bool argument to determine if convolutional block is used.
            res_block: bool argument to determine if residual block is used.
            dropout_rate: faction of the input units to drop.

        Examples::

            # for single channel input 4-channel output with patch size of (96,96,96), feature size of 32 and batch norm
            >>> net = UNETR(in_channels=1, out_channels=4, img_size=(96,96,96), feature_size=32, norm_name='batch')

            # for 4-channel input 3-channel output with patch size of (128,128,128), conv position embedding and instance norm
            >>> net = UNETR(in_channels=4, out_channels=3, img_size=(128,128,128), pos_embed='conv', norm_name='instance')

        """

        super().__init__()

        if not (0 <= dropout_rate <= 1):
            raise AssertionError("dropout_rate should be between 0 and 1.")

        if hidden_size % num_heads != 0:
            raise AssertionError("hidden size should be divisible by num_heads.")

        if pos_embed not in ["conv", "perceptron"]:
            raise KeyError(f"Position embedding layer of type {pos_embed} is not supported.")

        self.sigmoid = nn.Sigmoid()
        self.num_layers = 12
        self.patch_size = (feature_size, feature_size, img_size[2])
        self.feat_size = (
            img_size[0] // self.patch_size[0],
            img_size[1] // self.patch_size[1],
            img_size[2] // self.patch_size[2],
        )
        self.hidden_size = hidden_size
        self.classification = False
        self.vit = ViT(
            in_channels=in_channels,
            img_size=img_size,
            patch_size=self.patch_size,
            hidden_size=hidden_size,
            mlp_dim=mlp_dim,
            num_layers=self.num_layers,
            num_heads=num_heads,
            pos_embed=pos_embed,
            classification=self.classification,
            dropout_rate=dropout_rate,
        )
        self.encoder1 = UnetrBasicBlock(
            spatial_dims=3,
            in_channels=in_channels,
            out_channels=feature_size,
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.encoder2 = UnetrPrUpBlock(
            spatial_dims=3,
            in_channels=hidden_size,
            out_channels=feature_size * 2,
            num_layer=2,
            kernel_size=3,
            stride=1,
            upsample_kernel_size=[2,2,2],
            norm_name=norm_name,
            conv_block=conv_block,
            res_block=res_block,
        )
        self.encoder3 = UnetrPrUpBlock(
            spatial_dims=3,
            in_channels=hidden_size,
            out_channels=feature_size * 4,
            num_layer=1,
            kernel_size=3,
            stride=1,
            upsample_kernel_size=[2,2,2],
            norm_name=norm_name,
            conv_block=conv_block,
            res_block=res_block,
        )
        self.encoder4 = UnetrPrUpBlock(
            spatial_dims=3,
            in_channels=hidden_size,
            out_channels=feature_size * 8,
            num_layer=0,
            kernel_size=3,
            stride=1,
            upsample_kernel_size=[2,2,2],
            norm_name=norm_name,
            conv_block=conv_block,
            res_block=res_block,
        )
        self.decoder5 = UnetrUpBlock(
            spatial_dims=3,
            in_channels=hidden_size,
            out_channels=feature_size * 8,
            kernel_size=3,
            upsample_kernel_size=[2,2,2],
            norm_name=norm_name,
            res_block=res_block,
        )
        self.decoder4 = UnetrUpBlock(
            spatial_dims=3,
            in_channels=feature_size * 8,
            out_channels=feature_size * 4,
            kernel_size=3,
            upsample_kernel_size=[2,2,2],
            norm_name=norm_name,
            res_block=res_block,
        )
        self.decoder3 = UnetrUpBlock(
            spatial_dims=3,
            in_channels=feature_size * 4,
            out_channels=feature_size*2,
            kernel_size=3,   
            upsample_kernel_size=[2,2,2],
            norm_name=norm_name,
            res_block=res_block,
        )
        self.decoder2 = UnetrUpBlock(
            spatial_dims=3,
            in_channels=feature_size * 2,
            out_channels=feature_size,
            kernel_size=3,
            upsample_kernel_size=[2,4,4],
            norm_name=norm_name,
            res_block=res_block,
        )
        self.out = UnetOutBlock(spatial_dims=3, in_channels=feature_size, out_channels=out_channels)  # type: ignore

    def proj_feat(self, x, hidden_size, feat_size):
        x = x.view(x.size(0), feat_size[0], feat_size[1], feat_size[2], hidden_size)
        x = x.permute(0, 4, 3, 1, 2).contiguous()
        return x

    def load_from(self, weights):
        with torch.no_grad():
            res_weight = weights
            # copy weights from patch embedding
            for i in weights["state_dict"]:
                print(i)
            self.vit.patch_embedding.position_embeddings.copy_(
                weights["state_dict"]["module.transformer.patch_embedding.position_embeddings_3d"]
            )
            self.vit.patch_embedding.cls_token.copy_(
                weights["state_dict"]["module.transformer.patch_embedding.cls_token"]
            )
            self.vit.patch_embedding.patch_embeddings[1].weight.copy_(
                weights["state_dict"]["module.transformer.patch_embedding.patch_embeddings.1.weight"]
            )
            self.vit.patch_embedding.patch_embeddings[1].bias.copy_(
                weights["state_dict"]["module.transformer.patch_embedding.patch_embeddings.1.bias"]
            )

            # copy weights from  encoding blocks (default: num of blocks: 12)
            for bname, block in self.vit.blocks.named_children():
                block.loadFrom(weights, n_block=bname)
            # last norm layer of transformer
            self.vit.norm.weight.copy_(weights["state_dict"]["module.transformer.norm.weight"])
            self.vit.norm.bias.copy_(weights["state_dict"]["module.transformer.norm.bias"])

    def forward(self, x_in):
        #print("x_in:",x_in.shape)
        x, hidden_states_out = self.vit(x_in)
        #print("x",x.shape)
        enc1 = self.encoder1(x_in.permute(0, 1, 4, 2, 3).contiguous())
        #print("enc1",enc1.shape)
        x2 = hidden_states_out[3]
        #print("x2",x2.shape)
        #print("proj_x2",self.proj_feat(x2, self.hidden_size, self.feat_size).shape)
        enc2 = self.encoder2(self.proj_feat(x2, self.hidden_size, self.feat_size))
        #print("enc2",enc2.shape)
        x3 = hidden_states_out[6]
        #print("x3",x3.shape)
        #print("proj_x3",self.proj_feat(x3, self.hidden_size, self.feat_size).shape)
        enc3 = self.encoder3(self.proj_feat(x3, self.hidden_size, self.feat_size))
        #print("enc3",enc3.shape)
        x4 = hidden_states_out[9]
        #print("x4",x4.shape)
        #print("proj_x4",self.proj_feat(x4, self.hidden_size, self.feat_size).shape)
        enc4 = self.encoder4(self.proj_feat(x4, self.hidden_size, self.feat_size))
        #print("enc4",enc4.shape)
        dec4 = self.proj_feat(x, self.hidden_size, self.feat_size)
        #print("dec4",dec4.shape)
        dec3 = self.decoder5(dec4, enc4)
        #print("dec3",dec3.shape)
        dec2 = self.decoder4(dec3, enc3)
        #print("dec2",dec2.shape)
        dec1 = self.decoder3(dec2, enc2)
        #print("dec1",dec1.shape)
        out = self.decoder2(dec1, enc1)
        #print("out",out.shape)
        logits = self.out(out)
        #print('logits:',logits.shape)

        return logits

class CNN3D(nn.Module):
    def __init__(self, input_channels, output_channels):
        super(CNN3D, self).__init__()

        self.input_channels = input_channels
        self.output_channels = output_channels
    
     # Encoder layers

        self.encoder_conv_00 = nn.Sequential(*[nn.Conv3d(in_channels=self.input_channels,out_channels=64,kernel_size=7,padding=3)])
        self.encoder_conv_01 = nn.Sequential(*[nn.Conv3d(in_channels=64,out_channels=64,kernel_size=7,padding=3)])

        self.encoder_conv_10 = nn.Sequential(*[nn.Conv3d(in_channels=64,out_channels=128,kernel_size=7,padding=3)])
        self.encoder_conv_11 = nn.Sequential(*[nn.Conv3d(in_channels=128,out_channels=128,kernel_size=7,padding=3)])

        self.encoder_conv_20 = nn.Sequential(*[nn.Conv3d(in_channels=128,out_channels=256,kernel_size=3,padding=1)])
        self.encoder_conv_21 = nn.Sequential(*[nn.Conv3d(in_channels=256,out_channels=256,kernel_size=3,padding=1)])
        self.encoder_conv_22 = nn.Sequential(*[nn.Conv3d(in_channels=256,out_channels=256,kernel_size=3,padding=1)])

        self.encoder_conv_30 = nn.Sequential(*[nn.Conv3d(in_channels=256,out_channels=512,kernel_size=3,padding=1)])
        self.encoder_conv_31 = nn.Sequential(*[nn.Conv3d(in_channels=512,out_channels=512,kernel_size=3,padding=1)])
        self.encoder_conv_32 = nn.Sequential(*[nn.Conv3d(in_channels=512,out_channels=512,kernel_size=3,padding=1)])

        self.encoder_conv_40 = nn.Sequential(*[nn.Conv3d(in_channels=512,out_channels=512,kernel_size=3,padding=1)])
        self.encoder_conv_41 = nn.Sequential(*[nn.Conv3d(in_channels=512,out_channels=512,kernel_size=3,padding=1)])
        self.encoder_conv_42 = nn.Sequential(*[nn.Conv3d(in_channels=512,out_channels=512,kernel_size=3,padding=1)])


        # Decoder layers

        self.decoder_convtr_42 = nn.Sequential(*[nn.ConvTranspose3d(in_channels=512,out_channels=512,kernel_size=3,padding=1)])
        self.decoder_convtr_41 = nn.Sequential(*[nn.ConvTranspose3d(in_channels=512,out_channels=512,kernel_size=3,padding=1)])
        self.decoder_convtr_40 = nn.Sequential(*[nn.ConvTranspose3d(in_channels=512,out_channels=512,kernel_size=3,padding=1)])

        self.decoder_convtr_32 = nn.Sequential(*[nn.ConvTranspose3d(in_channels=512,out_channels=512,kernel_size=3,padding=1)])
        self.decoder_convtr_31 = nn.Sequential(*[nn.ConvTranspose3d(in_channels=512,out_channels=512,kernel_size=3,padding=1)])
        self.decoder_convtr_30 = nn.Sequential(*[nn.ConvTranspose3d(in_channels=512,out_channels=256,kernel_size=3,padding=1)])

        self.decoder_convtr_22 = nn.Sequential(*[nn.ConvTranspose3d(in_channels=256,out_channels=256,kernel_size=3,padding=1)])
        self.decoder_convtr_21 = nn.Sequential(*[nn.ConvTranspose3d(in_channels=256,out_channels=256,kernel_size=3,padding=1)])
        self.decoder_convtr_20 = nn.Sequential(*[nn.ConvTranspose3d(in_channels=256,out_channels=128,kernel_size=3,padding=1)])

        self.decoder_convtr_11 = nn.Sequential(*[nn.ConvTranspose3d(in_channels=128,out_channels=128,kernel_size=3,padding=1)])
        self.decoder_convtr_10 = nn.Sequential(*[nn.ConvTranspose3d(in_channels=128,out_channels=64,kernel_size=3,padding=1)])

        self.decoder_convtr_01 = nn.Sequential(*[nn.ConvTranspose3d(in_channels=64,out_channels=64,kernel_size=3,padding=1)])
        self.decoder_convtr_00 = nn.Sequential(*[nn.ConvTranspose3d(in_channels=64,out_channels=self.output_channels,kernel_size=3,padding=1)])

        self.dropout = nn.Dropout3d()

        self.sigmoid = nn.Sigmoid()

    def forward(self, input_img):

        # Encoder Stage - 1
        dim_0 = input_img.size()
        x_00 = F.relu(self.encoder_conv_00(input_img))
        x_01 = F.relu(self.encoder_conv_01(x_00))

        kernels = calc_kernel(x_01.size()[2:], kernel_size=2, depth=5)
        x_0, indices_0 = F.max_pool3d(x_01, kernel_size=kernels[0], stride=2, return_indices=True)

        # Encoder Stage - 2
        dim_1 = x_0.size()
        x_10 = F.relu(self.encoder_conv_10(x_0))
        x_11 = F.relu(self.encoder_conv_11(x_10))
        x_1, indices_1 = F.max_pool3d(x_11, kernel_size=kernels[1], stride=2, return_indices=True)

        x_1 = self.dropout(x_1)

        # Encoder Stage - 3
        dim_2 = x_1.size()
        x_20 = F.relu(self.encoder_conv_20(x_1))
        x_21 = F.relu(self.encoder_conv_21(x_20))
        x_22 = F.relu(self.encoder_conv_22(x_21))
        x_2, indices_2 = F.max_pool3d(x_22, kernel_size=kernels[2], stride=2, return_indices=True)

        # Encoder Stage - 4
        dim_3 = x_2.size()
        x_30 = F.relu(self.encoder_conv_30(x_2))
        x_31 = F.relu(self.encoder_conv_31(x_30))
        x_32 = F.relu(self.encoder_conv_32(x_31))
        x_3, indices_3 = F.max_pool3d(x_32, kernel_size=kernels[3], stride=2, return_indices=True)
        
        x_3 = self.dropout(x_3)
        
        # Encoder Stage - 5
        dim_4 = x_3.size()
        x_40 = F.relu(self.encoder_conv_40(x_3))
        x_41 = F.relu(self.encoder_conv_41(x_40))
        x_42 = F.relu(self.encoder_conv_42(x_41))
        x_4, indices_4 = F.max_pool3d(x_42, kernel_size=kernels[4], stride=2, return_indices=True)
        
        # Decoder

        # Decoder Stage - 5
        x_4d = F.max_unpool3d(x_4, indices_4, kernel_size=kernels[4], stride=2, output_size=dim_4)
        x_42d = F.relu(self.decoder_convtr_42(x_4d))
        x_41d = F.relu(self.decoder_convtr_41(x_42d))
        x_40d = F.relu(self.decoder_convtr_40(x_41d))

        x_40d = x_40d + x_3
        # x_40d = torch.cat((x_40d, x_3), dim=1)
        # c_40d = nn.Sequential(*[nn.Conv3d(in_channels=x_40d.size()[1],out_channels=indices_3.size()[1],kernel_size=3,padding=1)]).to(x_40d.device)
        # x_40d = F.relu(c_40d(x_40d))

        # Decoder Stage - 4
        x_3d = F.max_unpool3d(x_40d, indices_3, kernel_size=kernels[3], stride=2, output_size=dim_3)
        x_32d = F.relu(self.decoder_convtr_32(x_3d))
        x_31d = F.relu(self.decoder_convtr_31(x_32d))
        x_30d = self.dropout(F.relu(self.decoder_convtr_30(x_31d)))

        x_30d = x_30d + x_2
        # x_30d = torch.cat((x_30d, x_2), dim=1)
        # c_30d = nn.Sequential(*[nn.Conv3d(in_channels=x_30d.size()[1],out_channels=indices_2.size()[1],kernel_size=3,padding=1)]).to(x_30d.device)
        # x_30d = F.relu(c_30d(x_30d))

        # Decoder Stage - 3
        x_2d = F.max_unpool3d(x_30d, indices_2, kernel_size=kernels[2], stride=2, output_size=dim_2)
        x_22d = F.relu(self.decoder_convtr_22(x_2d))
        x_21d = F.relu(self.decoder_convtr_21(x_22d))
        x_20d = F.relu(self.decoder_convtr_20(x_21d))

        x_20d = x_20d + x_1
        # x_20d = torch.cat((x_20d, x_1), dim=1)
        # c_20d = nn.Sequential(*[nn.Conv3d(in_channels=x_20d.size()[1],out_channels=indices_1.size()[1],kernel_size=3,padding=1)]).to(x_20d.device)
        # x_20d = F.relu(c_20d(x_20d))

        # Decoder Stage - 2
        x_1d = F.max_unpool3d(x_20d, indices_1, kernel_size=kernels[1], stride=2, output_size=dim_1)
        x_11d = F.relu(self.decoder_convtr_11(x_1d))
        x_10d = self.dropout(F.relu(self.decoder_convtr_10(x_11d)))


        x_10d = x_10d + x_0
        # x_10d = torch.cat((x_10d, x_0), dim=1)
        # c_10d = nn.Sequential(*[nn.Conv3d(in_channels=x_10d.size()[1],out_channels=indices_0.size()[1],kernel_size=3,padding=1)]).to(x_10d.device)
        # x_10d = F.relu(c_10d(x_10d))

        # Decoder Stage - 1
        x_0d = F.max_unpool3d(x_10d, indices_0, kernel_size=kernels[0], stride=2, output_size=dim_0)

        x_01d = F.relu(self.decoder_convtr_01(x_0d))
        x_00d = self.decoder_convtr_00(x_01d)

        return self.sigmoid(x_00d)

class Maike_CNN3D(nn.Module):
    def __init__(self, input_channels, output_channels):
        super(CNN3D, self).__init__()

        self.input_channels = input_channels
        self.output_channels = output_channels
     # Encoder layers

        self.encoder_conv_00 = nn.Sequential(*[nn.Conv3d(in_channels=self.input_channels,out_channels=64,kernel_size=3,padding=1)])
        self.encoder_conv_01 = nn.Sequential(*[nn.Conv3d(in_channels=64,out_channels=64,kernel_size=3,padding=1)])

        self.encoder_conv_10 = nn.Sequential(*[nn.Conv3d(in_channels=64,out_channels=128,kernel_size=3,padding=1)])
        self.encoder_conv_11 = nn.Sequential(*[nn.Conv3d(in_channels=128,out_channels=128,kernel_size=3,padding=1)])

        self.encoder_conv_20 = nn.Sequential(*[nn.Conv3d(in_channels=128,out_channels=256,kernel_size=3,padding=1)])
        self.encoder_conv_21 = nn.Sequential(*[nn.Conv3d(in_channels=256,out_channels=256,kernel_size=3,padding=1)])
        self.encoder_conv_22 = nn.Sequential(*[nn.Conv3d(in_channels=256,out_channels=256,kernel_size=3,padding=1)])

        self.encoder_conv_30 = nn.Sequential(*[nn.Conv3d(in_channels=256,out_channels=512,kernel_size=3,padding=1)])
        self.encoder_conv_31 = nn.Sequential(*[nn.Conv3d(in_channels=512,out_channels=512,kernel_size=3,padding=1)])
        self.encoder_conv_32 = nn.Sequential(*[nn.Conv3d(in_channels=512,out_channels=512,kernel_size=3,padding=1)])

        self.encoder_conv_40 = nn.Sequential(*[nn.Conv3d(in_channels=512,out_channels=512,kernel_size=3,padding=1)])
        self.encoder_conv_41 = nn.Sequential(*[nn.Conv3d(in_channels=512,out_channels=512,kernel_size=3,padding=1)])
        self.encoder_conv_42 = nn.Sequential(*[nn.Conv3d(in_channels=512,out_channels=512,kernel_size=3,padding=1)])

        # self.encoder_conv_50 = nn.Sequential(*[nn.Conv3d(in_channels=1024,out_channels=1024,kernel_size=3,padding=1)])
        # self.encoder_conv_51 = nn.Sequential(*[nn.Conv3d(in_channels=1024,out_channels=1024,kernel_size=3,padding=1)])
        # self.encoder_conv_52 = nn.Sequential(*[nn.Conv3d(in_channels=1024,out_channels=1024,kernel_size=3,padding=1)])

        # Decoder layers

        # self.decoder_convtr_52 = nn.Sequential(*[nn.ConvTranspose3d(in_channels=1024,out_channels=1024,kernel_size=3,padding=1)])
        # self.decoder_convtr_51 = nn.Sequential(*[nn.ConvTranspose3d(in_channels=1024,out_channels=1024,kernel_size=3,padding=1)])
        # self.decoder_convtr_50 = nn.Sequential(*[nn.ConvTranspose3d(in_channels=1024,out_channels=1024,kernel_size=3,padding=1)])

        self.decoder_convtr_42 = nn.Sequential(*[nn.ConvTranspose3d(in_channels=512,out_channels=512,kernel_size=3,padding=1)])
        self.decoder_convtr_41 = nn.Sequential(*[nn.ConvTranspose3d(in_channels=512,out_channels=512,kernel_size=3,padding=1)])
        self.decoder_convtr_40 = nn.Sequential(*[nn.ConvTranspose3d(in_channels=512,out_channels=512,kernel_size=3,padding=1)])

        self.decoder_convtr_32 = nn.Sequential(*[nn.ConvTranspose3d(in_channels=512,out_channels=512,kernel_size=3,padding=1)])
        self.decoder_convtr_31 = nn.Sequential(*[nn.ConvTranspose3d(in_channels=512,out_channels=512,kernel_size=3,padding=1)])
        self.decoder_convtr_30 = nn.Sequential(*[nn.ConvTranspose3d(in_channels=512,out_channels=256,kernel_size=3,padding=1)])

        self.decoder_convtr_22 = nn.Sequential(*[nn.ConvTranspose3d(in_channels=256,out_channels=256,kernel_size=3,padding=1)])
        self.decoder_convtr_21 = nn.Sequential(*[nn.ConvTranspose3d(in_channels=256,out_channels=256,kernel_size=3,padding=1)])
        self.decoder_convtr_20 = nn.Sequential(*[nn.ConvTranspose3d(in_channels=256,out_channels=128,kernel_size=3,padding=1)])

        self.decoder_convtr_11 = nn.Sequential(*[nn.ConvTranspose3d(in_channels=128,out_channels=128,kernel_size=3,padding=1)])
        self.decoder_convtr_10 = nn.Sequential(*[nn.ConvTranspose3d(in_channels=128,out_channels=64,kernel_size=3,padding=1)])

        self.decoder_convtr_01 = nn.Sequential(*[nn.ConvTranspose3d(in_channels=64,out_channels=64,kernel_size=3,padding=1)])
        self.decoder_convtr_00 = nn.Sequential(*[nn.ConvTranspose3d(in_channels=64,out_channels=self.output_channels,kernel_size=3,padding=1)])
        
        self.dropout = nn.Dropout3d()

        self.sigmoid = nn.Sigmoid()

    def forward(self, input_img):

        # Encoder Stage - 1
        dim_0 = input_img.size()

        x_00 = F.relu(self.encoder_conv_00(input_img))
        x_01 = F.relu(self.encoder_conv_01(x_00))

        kernels = calc_kernel(x_01.size()[2:], kernel_size=2, depth=5)
        x_0, indices_0 = F.max_pool3d(x_01, kernel_size=kernels[0], stride=2, return_indices=True)

        # Encoder Stage - 2
        dim_1 = x_0.size()
        x_10 = F.relu(self.encoder_conv_10(x_0))
        x_11 = F.relu(self.encoder_conv_11(x_10))

        res_0 = F.interpolate(x_0, size=x_11.size()[2:])
        cres_0 = nn.Sequential(*[nn.Conv3d(in_channels=x_0.size()[1],out_channels=x_11.size()[1],kernel_size=1,padding=0)]).to(res_0.device)
        res_0 = cres_0(res_0)
        x_11 = x_11 + res_0

        x_11 = self.dropout(x_11)

        x_1, indices_1 = F.max_pool3d(x_11, kernel_size=kernels[1], stride=2, return_indices=True)

        # Encoder Stage - 3
        dim_2 = x_1.size()
        x_20 = F.relu(self.encoder_conv_20(x_1))
        x_21 = F.relu(self.encoder_conv_21(x_20))
        x_22 = F.relu(self.encoder_conv_22(x_21))

        res_1 = F.interpolate(x_1, size=x_22.size()[2:])
        cres_1 = nn.Sequential(*[nn.Conv3d(in_channels=x_1.size()[1],out_channels=x_22.size()[1],kernel_size=1,padding=0)]).to(res_1.device)
        res_1 = cres_1(res_1)
        x_22 = x_22 + res_1

        x_22 = self.dropout(x_22)

        x_2, indices_2 = F.max_pool3d(x_22, kernel_size=kernels[2], stride=2, return_indices=True)

        # Encoder Stage - 4
        dim_3 = x_2.size()
        x_30 = F.relu(self.encoder_conv_30(x_2))
        x_31 = F.relu(self.encoder_conv_31(x_30))
        x_32 = F.relu(self.encoder_conv_32(x_31))

        res_2 = F.interpolate(x_2, size=x_32.size()[2:])
        cres_2 = nn.Sequential(*[nn.Conv3d(in_channels=x_2.size()[1],out_channels=x_32.size()[1],kernel_size=1,padding=0)]).to(res_2.device)
        res_2 = cres_2(res_2)
        x_32 = x_32 + res_2

        x_32 = self.dropout(x_32)

        x_3, indices_3 = F.max_pool3d(x_32, kernel_size=kernels[3], stride=2, return_indices=True)
        
        # Encoder Stage - 5
        dim_4 = x_3.size()
        x_40 = F.relu(self.encoder_conv_40(x_3))
        x_41 = F.relu(self.encoder_conv_41(x_40))
        x_42 = F.relu(self.encoder_conv_42(x_41))

        res_3 = F.interpolate(x_3, size=x_42.size()[2:])
        cres_3 = nn.Sequential(*[nn.Conv3d(in_channels=x_3.size()[1],out_channels=x_42.size()[1],kernel_size=1,padding=0)]).to(res_3.device)
        res_3 = cres_3(res_3)
        x_42 = x_42 + res_3

        x_42 = self.dropout(x_42)

        x_4, indices_4 = F.max_pool3d(x_42, kernel_size=kernels[4], stride=2, return_indices=True)

        # Encoder Stage - 6
        # dim_5 = x_4.size()
        # x_50 = F.relu(self.encoder_conv_50(x_4))
        # x_51 = F.relu(self.encoder_conv_51(x_50))
        # x_52 = F.relu(self.encoder_conv_52(x_51))
        # x_5, indices_5 = F.max_pool3d(x_52, kernel_size=kernels[5], stride=2, return_indices=True)
        # res_4 = F.interpolate(x_4, size=x_5.size()[2:])
        # x_5 = x_5 + res_4

        # Decoder

        # Decoder Stage - 6
        
        # x_5d = F.max_unpool3d(x_5, indices_5, kernel_size=kernels[5], stride=2, output_size=dim_5)
        # x_52d = F.relu(self.decoder_convtr_52(x_5d))
        # x_51d = F.relu(self.decoder_convtr_51(x_52d))

        # res_5d = F.interpolate(x_5d, size=x_5.size()[2:])

        # x_50d = F.relu(self.decoder_convtr_50(x_51d))
        
        # x_50d = x_50d + x_4
        # x_50d = torch.cat((x_50d, x_4), dim=1)
        # c_50d = nn.Sequential(*[nn.Conv3d(in_channels=x_50d.size()[1],out_channels=indices_4.size()[1],kernel_size=3,padding=1)]).to(x_50d.device)
        # x_50d = c_50d(x_50d)

        # Decoder Stage - 5
        x_4d = F.max_unpool3d(x_4, indices_4, kernel_size=kernels[4], stride=2, output_size=dim_4)
        x_4d = torch.cat((x_4d, x_3), dim=1)

        c_4d = nn.Sequential(*[nn.Conv3d(in_channels=x_4d.size()[1],out_channels=indices_4.size()[1],kernel_size=3,padding=1)]).to(x_4d.device)
        x_4d = c_4d(x_4d)

        x_42d = F.relu(self.decoder_convtr_42(x_4d))
        x_41d = F.relu(self.decoder_convtr_41(x_42d))
        x_40d = F.relu(self.decoder_convtr_40(x_41d))

        res_4d = F.interpolate(x_4d, size=x_40d.size()[2:])
        cres_4d = nn.Sequential(*[nn.Conv3d(in_channels=x_4d.size()[1],out_channels=x_40d.size()[1],kernel_size=1,padding=0)]).to(x_4d.device)
        res_4d = cres_4d(res_4d)
        x_40d = x_40d + res_4d

        # Decoder Stage - 4
        x_3d = F.max_unpool3d(x_40d, indices_3, kernel_size=kernels[3], stride=2, output_size=dim_3)
        x_3d = torch.cat((x_3d, x_2), dim=1)
        c_3d = nn.Sequential(*[nn.Conv3d(in_channels=x_3d.size()[1],out_channels=indices_3.size()[1],kernel_size=3,padding=1)]).to(x_3d.device)
        x_3d = c_3d(x_3d)

        x_32d = F.relu(self.decoder_convtr_32(x_3d))
        x_31d = F.relu(self.decoder_convtr_31(x_32d))
        x_30d = F.relu(self.decoder_convtr_30(x_31d))

        res_3d = F.interpolate(x_3d, size=x_30d.size()[2:])
        cres_3d = nn.Sequential(*[nn.Conv3d(in_channels=x_3d.size()[1],out_channels=x_30d.size()[1],kernel_size=1,padding=0)]).to(x_3d.device)
        res_3d = cres_3d(res_3d)
        x_30d = x_30d + res_3d

        # Decoder Stage - 3
        x_2d = F.max_unpool3d(x_30d, indices_2, kernel_size=kernels[2], stride=2, output_size=dim_2)
        x_2d = torch.cat((x_2d, x_1), dim=1)
        c_2d = nn.Sequential(*[nn.Conv3d(in_channels=x_2d.size()[1],out_channels=indices_2.size()[1],kernel_size=3,padding=1)]).to(x_2d.device)
        x_2d = c_2d(x_2d)

        x_22d = F.relu(self.decoder_convtr_22(x_2d))
        x_21d = F.relu(self.decoder_convtr_21(x_22d))
        x_20d = F.relu(self.decoder_convtr_20(x_21d))

        res_2d = F.interpolate(x_2d, size=x_20d.size()[2:])
        cres_2d = nn.Sequential(*[nn.Conv3d(in_channels=x_2d.size()[1],out_channels=x_20d.size()[1],kernel_size=1,padding=0)]).to(x_2d.device)
        res_2d = cres_2d(res_2d)
        x_20d = x_20d + res_2d

        # Decoder Stage - 2
        x_1d = F.max_unpool3d(x_20d, indices_1, kernel_size=kernels[1], stride=2, output_size=dim_1)
        x_1d = torch.cat((x_1d, x_0), dim=1)
        c_1d = nn.Sequential(*[nn.Conv3d(in_channels=x_1d.size()[1],out_channels=indices_1.size()[1],kernel_size=3,padding=1)]).to(x_1d.device)
        x_1d = c_1d(x_1d)

        x_11d = F.relu(self.decoder_convtr_11(x_1d))
        x_10d = F.relu(self.decoder_convtr_10(x_11d))

        res_1d = F.interpolate(x_1d, size=x_10d.size()[2:])
        cres_1d = nn.Sequential(*[nn.Conv3d(in_channels=x_1d.size()[1],out_channels=x_10d.size()[1],kernel_size=1,padding=0)]).to(x_1d.device)
        res_1d = cres_1d(res_1d)
        x_10d = x_10d + res_1d

        # Decoder Stage - 1
        x_0d = F.max_unpool3d(x_10d, indices_0, kernel_size=kernels[0], stride=2, output_size=dim_0)

        x_01d = F.relu(self.decoder_convtr_01(x_0d))
        x_00d = self.decoder_convtr_00(x_01d)

        return self.sigmoid(x_00d)

class FNet(nn.Module):
    """ Optical flow estimation network
    """

    def __init__(self, in_nc):
        super(FNet, self).__init__()

        self.encoder1 = nn.Sequential(
            nn.Conv2d(2*in_nc, 32, 3, 1, 1, bias=True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(32, 32, 3, 1, 1, bias=True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.MaxPool2d(2, 2))

        self.encoder2 = nn.Sequential(
            nn.Conv2d(32, 64, 3, 1, 1, bias=True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 64, 3, 1, 1, bias=True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.MaxPool2d(2, 2))

        self.encoder3 = nn.Sequential(
            nn.Conv2d(64, 128, 3, 1, 1, bias=True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 128, 3, 1, 1, bias=True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.MaxPool2d(2, 2))

        self.decoder1 = nn.Sequential(
            nn.Conv2d(128, 256, 3, 1, 1, bias=True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 256, 3, 1, 1, bias=True),
            nn.LeakyReLU(0.2, inplace=True))

        self.decoder2 = nn.Sequential(
            nn.Conv2d(256, 128, 3, 1, 1, bias=True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 128, 3, 1, 1, bias=True),
            nn.LeakyReLU(0.2, inplace=True))

        self.decoder3 = nn.Sequential(
            nn.Conv2d(128, 64, 3, 1, 1, bias=True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 64, 3, 1, 1, bias=True),
            nn.LeakyReLU(0.2, inplace=True))

        self.flow = nn.Sequential(
            nn.Conv2d(64, 32, 3, 1, 1, bias=True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(32, 2, 3, 1, 1, bias=True))

    def forward(self, x1, x2):
        """ Compute optical flow from x1 to x2
        """

        out = self.encoder1(torch.cat([x1, x2], dim=1))
        out = self.encoder2(out)
        out = self.encoder3(out)
        out = F.interpolate(
            self.decoder1(out), scale_factor=2, mode='bilinear', align_corners=False)
        out = F.interpolate(
            self.decoder2(out), scale_factor=2, mode='bilinear', align_corners=False)
        out = F.interpolate(
            self.decoder3(out), scale_factor=2, mode='bilinear', align_corners=False)
        out = torch.tanh(self.flow(out)) * 24  # 24 is the max velocity

        return out
    

class CNN2D(nn.Module):
    def __init__(self, input_channels, output_channels):
        super(CNN2D, self).__init__()

        self.input_channels = input_channels
        self.output_channels = output_channels

        # Encoder layers

        self.encoder_conv_00 = nn.Sequential(*[nn.Conv2d(in_channels=self.input_channels,out_channels=64,kernel_size=3,padding=1)])
        self.encoder_conv_01 = nn.Sequential(*[nn.Conv2d(in_channels=64,out_channels=64,kernel_size=3,padding=1)])

        self.encoder_conv_10 = nn.Sequential(*[nn.Conv2d(in_channels=64,out_channels=128,kernel_size=3,padding=1)])
        self.encoder_conv_11 = nn.Sequential(*[nn.Conv2d(in_channels=128,out_channels=128,kernel_size=3,padding=1)])

        self.encoder_conv_20 = nn.Sequential(*[nn.Conv2d(in_channels=128,out_channels=256,kernel_size=3,padding=1)])
        self.encoder_conv_21 = nn.Sequential(*[nn.Conv2d(in_channels=256,out_channels=256,kernel_size=3,padding=1)])
        self.encoder_conv_22 = nn.Sequential(*[nn.Conv2d(in_channels=256,out_channels=256,kernel_size=3,padding=1)])

        self.encoder_conv_30 = nn.Sequential(*[nn.Conv2d(in_channels=256,out_channels=512,kernel_size=3,padding=1)])
        self.encoder_conv_31 = nn.Sequential(*[nn.Conv2d(in_channels=512,out_channels=512,kernel_size=3,padding=1)])
        self.encoder_conv_32 = nn.Sequential(*[nn.Conv2d(in_channels=512,out_channels=512,kernel_size=3,padding=1)])

        self.encoder_conv_40 = nn.Sequential(*[nn.Conv2d(in_channels=512,out_channels=512,kernel_size=3,padding=1)])
        self.encoder_conv_41 = nn.Sequential(*[nn.Conv2d(in_channels=512,out_channels=512,kernel_size=3,padding=1)])
        self.encoder_conv_42 = nn.Sequential(*[nn.Conv2d(in_channels=512,out_channels=512,kernel_size=3,padding=1)])


        # Decoder layers

        self.decoder_convtr_42 = nn.Sequential(*[nn.ConvTranspose2d(in_channels=512,out_channels=512,kernel_size=3,padding=1)])
        self.decoder_convtr_41 = nn.Sequential(*[nn.ConvTranspose2d(in_channels=512,out_channels=512,kernel_size=3,padding=1)])
        self.decoder_convtr_40 = nn.Sequential(*[nn.ConvTranspose2d(in_channels=512,out_channels=512,kernel_size=3,padding=1)])

        self.decoder_convtr_32 = nn.Sequential(*[nn.ConvTranspose2d(in_channels=512,out_channels=512,kernel_size=3,padding=1)])
        self.decoder_convtr_31 = nn.Sequential(*[nn.ConvTranspose2d(in_channels=512,out_channels=512,kernel_size=3,padding=1)])
        self.decoder_convtr_30 = nn.Sequential(*[nn.ConvTranspose2d(in_channels=512,out_channels=256,kernel_size=3,padding=1)])

        self.decoder_convtr_22 = nn.Sequential(*[nn.ConvTranspose2d(in_channels=256,out_channels=256,kernel_size=3,padding=1)])
        self.decoder_convtr_21 = nn.Sequential(*[nn.ConvTranspose2d(in_channels=256,out_channels=256,kernel_size=3,padding=1)])
        self.decoder_convtr_20 = nn.Sequential(*[nn.ConvTranspose2d(in_channels=256,out_channels=128,kernel_size=3,padding=1)])

        self.decoder_convtr_11 = nn.Sequential(*[nn.ConvTranspose2d(in_channels=128,out_channels=128,kernel_size=3,padding=1)])
        self.decoder_convtr_10 = nn.Sequential(*[nn.ConvTranspose2d(in_channels=128,out_channels=64,kernel_size=3,padding=1)])

        self.decoder_convtr_01 = nn.Sequential(*[nn.ConvTranspose2d(in_channels=64,out_channels=64,kernel_size=3,padding=1)])
        self.decoder_convtr_00 = nn.Sequential(*[nn.ConvTranspose2d(in_channels=64,out_channels=self.output_channels,kernel_size=3,padding=1)])

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
    
class UNETR_2(nn.Module):
    """
    UNETR based on: "Hatamizadeh et al.,
    UNETR: Transformers for 3D Medical Image Segmentation <https://arxiv.org/abs/2103.10504>"
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        img_size: Tuple[int, int, int],
        feature_size: int = 16,
        hidden_size: int = 768,
        mlp_dim: int = 3072,
        num_heads: int = 12,
        pos_embed: str = "perceptron",
        norm_name: Union[Tuple, str] = "instance",
        conv_block: bool = False,
        res_block: bool = True,
        dropout_rate: float = 0.0,
    ) -> None:
        """
        Args:
            in_channels: dimension of input channels.
            out_channels: dimension of output channels.
            img_size: dimension of input image.
            feature_size: dimension of network feature size.
            hidden_size: dimension of hidden layer.
            mlp_dim: dimension of feedforward layer.
            num_heads: number of attention heads.
            pos_embed: position embedding layer type.
            norm_name: feature normalization type and arguments.
            conv_block: bool argument to determine if convolutional block is used.
            res_block: bool argument to determine if residual block is used.
            dropout_rate: faction of the input units to drop.

        Examples::

            # for single channel input 4-channel output with patch size of (96,96,96), feature size of 32 and batch norm
            >>> net = UNETR(in_channels=1, out_channels=4, img_size=(96,96,96), feature_size=32, norm_name='batch')

            # for 4-channel input 3-channel output with patch size of (128,128,128), conv position embedding and instance norm
            >>> net = UNETR(in_channels=4, out_channels=3, img_size=(128,128,128), pos_embed='conv', norm_name='instance')

        """

        super().__init__()

        if not (0 <= dropout_rate <= 1):
            raise AssertionError("dropout_rate should be between 0 and 1.")

        if hidden_size % num_heads != 0:
            raise AssertionError("hidden size should be divisible by num_heads.")

        if pos_embed not in ["conv", "perceptron"]:
            raise KeyError(f"Position embedding layer of type {pos_embed} is not supported.")

        self.num_layers = 12
        self.patch_size = (16, 16, 8)
        self.feat_size = (
            img_size[0] // self.patch_size[0],
            img_size[1] // self.patch_size[1],
            img_size[2] // self.patch_size[2],
        )
        self.hidden_size = hidden_size
        self.classification = False
        self.vit = ViT(
            in_channels=in_channels,
            img_size=img_size,
            patch_size=self.patch_size,
            hidden_size=hidden_size,
            mlp_dim=mlp_dim,
            num_layers=self.num_layers,
            num_heads=num_heads,
            pos_embed=pos_embed,
            classification=self.classification,
            dropout_rate=dropout_rate,
        )
        self.encoder1 = UnetrBasicBlock(
            spatial_dims=3,
            in_channels=in_channels,
            out_channels=feature_size,
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.encoder2 = UnetrPrUpBlock(
            spatial_dims=3,
            in_channels=hidden_size,
            out_channels=feature_size * 2,
            num_layer=2,
            kernel_size=3,
            stride=1,
            upsample_kernel_size=[2,2,2],
            norm_name=norm_name,
            conv_block=conv_block,
            res_block=res_block,
        )
        self.encoder3 = UnetrPrUpBlock(
            spatial_dims=3,
            in_channels=hidden_size,
            out_channels=feature_size * 4,
            num_layer=1,
            kernel_size=3,
            stride=1,
            upsample_kernel_size=[2,2,2],
            norm_name=norm_name,
            conv_block=conv_block,
            res_block=res_block,
        )
        self.encoder4 = UnetrPrUpBlock(
            spatial_dims=3,
            in_channels=hidden_size,
            out_channels=feature_size * 8,
            num_layer=0,
            kernel_size=3,
            stride=1,
            upsample_kernel_size=[2,2,2],
            norm_name=norm_name,
            conv_block=conv_block,
            res_block=res_block,
        )
        self.decoder5 = UnetrUpBlock(
            spatial_dims=3,
            in_channels=hidden_size,
            out_channels=feature_size * 8,
            kernel_size=3,
            upsample_kernel_size=[2,2,2],
            norm_name=norm_name,
            res_block=res_block,
        )
        self.decoder4 = UnetrUpBlock(
            spatial_dims=3,
            in_channels=feature_size * 8,
            out_channels=feature_size * 4,
            kernel_size=3,
            upsample_kernel_size=[2,2,2],
            norm_name=norm_name,
            res_block=res_block,
        )
        self.decoder3 = UnetrUpBlock(
            spatial_dims=3,
            in_channels=feature_size * 4,
            out_channels=feature_size*2,
            kernel_size=3,   
            upsample_kernel_size=[2,2,2],
            norm_name=norm_name,
            res_block=res_block,
        )
        self.decoder2 = UnetrUpBlock(
            spatial_dims=3,
            in_channels=feature_size * 2,
            out_channels=feature_size,
            kernel_size=3,
            upsample_kernel_size=[1,2,2],
            norm_name=norm_name,
            res_block=res_block,
        )
        self.out = UnetOutBlock(spatial_dims=3, in_channels=feature_size, out_channels=out_channels)  # type: ignore

    def proj_feat(self, x, hidden_size, feat_size):
        x = x.view(x.size(0), feat_size[0], feat_size[1], feat_size[2], hidden_size)
        x = x.permute(0, 4, 3, 1, 2).contiguous()
        return x

    def load_from(self, weights):
        with torch.no_grad():
            res_weight = weights
            # copy weights from patch embedding
            for i in weights["state_dict"]:
                print(i)
            self.vit.patch_embedding.position_embeddings.copy_(
                weights["state_dict"]["module.transformer.patch_embedding.position_embeddings_3d"]
            )
            self.vit.patch_embedding.cls_token.copy_(
                weights["state_dict"]["module.transformer.patch_embedding.cls_token"]
            )
            self.vit.patch_embedding.patch_embeddings[1].weight.copy_(
                weights["state_dict"]["module.transformer.patch_embedding.patch_embeddings.1.weight"]
            )
            self.vit.patch_embedding.patch_embeddings[1].bias.copy_(
                weights["state_dict"]["module.transformer.patch_embedding.patch_embeddings.1.bias"]
            )

            # copy weights from  encoding blocks (default: num of blocks: 12)
            for bname, block in self.vit.blocks.named_children():
                print(block)
                block.loadFrom(weights, n_block=bname)
            # last norm layer of transformer
            self.vit.norm.weight.copy_(weights["state_dict"]["module.transformer.norm.weight"])
            self.vit.norm.bias.copy_(weights["state_dict"]["module.transformer.norm.bias"])

    def forward(self, x_in):
        #print("x_in:",x_in.shape)
        x, hidden_states_out = self.vit(x_in)
        #print("x",x.shape)
        enc1 = self.encoder1(x_in.permute(0, 1, 4, 2, 3).contiguous())
        #print("enc1",enc1.shape)
        x2 = hidden_states_out[3]
        #print("x2",x2.shape)
        #print("proj_x2",self.proj_feat(x2, self.hidden_size, self.feat_size).shape)
        enc2 = self.encoder2(self.proj_feat(x2, self.hidden_size, self.feat_size))
        #print("enc2",enc2.shape)
        x3 = hidden_states_out[6]
        #print("x3",x3.shape)
        #print("proj_x3",self.proj_feat(x3, self.hidden_size, self.feat_size).shape)
        enc3 = self.encoder3(self.proj_feat(x3, self.hidden_size, self.feat_size))
        #print("enc3",enc3.shape)
        x4 = hidden_states_out[9]
        #print("x4",x4.shape)
        #print("proj_x4",self.proj_feat(x4, self.hidden_size, self.feat_size).shape)
        enc4 = self.encoder4(self.proj_feat(x4, self.hidden_size, self.feat_size))
        #print("enc4",enc4.shape)
        dec4 = self.proj_feat(x, self.hidden_size, self.feat_size)
        #print("dec4",dec4.shape)
        dec3 = self.decoder5(dec4, enc4)
        #print("dec3",dec3.shape)
        dec2 = self.decoder4(dec3, enc3)
        #print("dec2",dec2.shape)
        dec1 = self.decoder3(dec2, enc2)
        #print("dec1",dec1.shape)
        out = self.decoder2(dec1, enc1)
        #print("out",out.shape)
        logits = self.out(out)
        #print('logits:',logits.shape)

        return logits

class ResUnetPlusPlus(nn.Module):
    def __init__(self, channel, filters=[32, 64, 128, 256, 512], drop_p=0.0):
        super(ResUnetPlusPlus, self).__init__()

        self.input_layer = nn.Sequential(
            nn.Conv3d(channel, filters[0], kernel_size=3, padding=1),
            # nn.BatchNorm3d(filters[0]),
            nn.GroupNorm(1, filters[0]),
            nn.ReLU(),
            nn.Dropout3d(drop_p),
            nn.Conv3d(filters[0], filters[0], kernel_size=3, padding=1),
        )
        self.input_skip = nn.Sequential(
            nn.Conv3d(channel, filters[0], kernel_size=3, padding=1)
        )

        self.squeeze_excite1 = Squeeze_Excite_Block(filters[0], drop_p=drop_p)

        self.residual_conv1 = ResidualConv(filters[0], filters[1], 2, 1, drop_p=drop_p)

        self.squeeze_excite2 = Squeeze_Excite_Block(filters[1], drop_p=drop_p)

        self.residual_conv2 = ResidualConv(filters[1], filters[2], 2, 1, drop_p=drop_p)

        self.squeeze_excite3 = Squeeze_Excite_Block(filters[2], drop_p=drop_p)

        self.residual_conv3 = ResidualConv(filters[2], filters[3], 2, 1, drop_p=drop_p)

        self.aspp_bridge = ASPP(filters[3], filters[4], drop_p=drop_p)

        self.attn1 = AttentionBlock(filters[2], filters[4], filters[4], drop_p=drop_p)
        self.upsample1 = Upsample_(2)
        self.up_residual_conv1 = ResidualConv(filters[4] + filters[2], filters[3], 1, 1, drop_p=drop_p)

        self.attn2 = AttentionBlock(filters[1], filters[3], filters[3], drop_p=drop_p)
        self.upsample2 = Upsample_(2)
        self.up_residual_conv2 = ResidualConv(filters[3] + filters[1], filters[2], 1, 1, drop_p=drop_p)

        self.attn3 = AttentionBlock(filters[0], filters[2], filters[2], drop_p=drop_p)
        self.upsample3 = Upsample_(2)
        self.up_residual_conv3 = ResidualConv(filters[2] + filters[0], filters[1], 1, 1, drop_p=drop_p)

        self.aspp_out = ASPP(filters[1], filters[0], drop_p=drop_p)

        self.output_layer = nn.Sequential(nn.Conv3d(filters[0], 1, 1), nn.Sigmoid())

    def forward(self, x):

        x1 = self.input_layer(x) + self.input_skip(x)

        x2 = self.squeeze_excite1(x1)
        x2 = self.residual_conv1(x2)

        x3 = self.squeeze_excite2(x2)
        x3 = self.residual_conv2(x3)

        x4 = self.squeeze_excite3(x3)
        x4 = self.residual_conv3(x4)

        x5 = self.aspp_bridge(x4)

        x6 = self.attn1(x3, x5)
        x6 = self.upsample1(x6)
        x6 = torch.cat([x6, x3], dim=1)
        x6 = self.up_residual_conv1(x6)

        x7 = self.attn2(x2, x6)
        x7 = self.upsample2(x7)
        x7 = torch.cat([x7, x2], dim=1)
        x7 = self.up_residual_conv2(x7)

        x8 = self.attn3(x1, x7)
        x8 = self.upsample3(x8)
        x8 = torch.cat([x8, x1], dim=1)
        x8 = self.up_residual_conv3(x8)

        x9 = self.aspp_out(x8)
        out = self.output_layer(x9)

        return out