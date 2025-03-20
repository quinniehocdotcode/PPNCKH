import math
import torch
import torch.nn as nn
from einops.layers.torch import Rearrange
import torch
import torch.nn as nn
import torch.nn.functional as F
device = torch.device("cuda")
class GELUConvBlock(nn.Module): # ứng dụng vào đây 
    def __init__(self, in_ch, out_ch, group_size):
        super().__init__()
        layers = [
            nn.Conv2d(in_ch, out_ch, 3, 1, 1),
            nn.GroupNorm(group_size, out_ch),
            nn.GELU(),
        ]
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class RearrangePoolBlock(nn.Module): # chức năng giống maxpoolingpooling
    def __init__(self, in_chs, group_size):
        super().__init__()
        self.rearrange = Rearrange("b c (h p1) (w p2) -> b (c p1 p2) h w", p1=2, p2=2)
        self.conv = GELUConvBlock(4 * in_chs, in_chs, group_size)

    def forward(self, x):
        x = self.rearrange(x)
        return self.conv(x)


# CrossAttention giữ nguyên
class CrossAttention(nn.Module):
    def __init__(self, dim, context_dim):
        super().__init__()
        self.to_q = nn.Linear(dim, dim, bias=False)
        self.to_k = nn.Linear(context_dim, dim, bias=False)
        self.to_v = nn.Linear(context_dim, dim, bias=False)
        self.scale = dim ** -0.5  

    def forward(self, x, context):
        q = self.to_q(x)
        k = self.to_k(context)
        v = self.to_v(context)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        out = attn @ v
        return out + x  # Residual Connection

# U-Net Encoder với Cross-Attention
class UNetEncoderWithCrossAttention(nn.Module):
    def __init__(self,in_ch, out_ch, cond_dim):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.cross_attn = CrossAttention(out_ch, cond_dim)
        self.out_ch = out_ch

    def forward(self, x, condition):
        B, C, H, W = x.shape  # Lưu H, W để reshape sau
        # Biến đổi thành (B, H*W, C)
        x = self.conv(x).flatten(2).transpose(1, 2)  
        # Áp dụng Cross-Attention
        x = self.cross_attn(x, condition.squeeze(1) )
        
        # Reshape về (B, C, H, W)
        x = x.transpose(1, 2).contiguous().view(B, self.out_ch, H, W)
        return x
#Remake khối DownBlock
class ReDownBlock(nn.Module):
    def __init__(self, in_chs, out_chs, dim_emdeding, group_size):
        super(ReDownBlock, self).__init__()
        self.encode1 = UNetEncoderWithCrossAttention(in_chs, out_chs, dim_emdeding)
        self.encode2 = UNetEncoderWithCrossAttention(out_chs, out_chs, dim_emdeding)
        self.RearrangePoolBlock = RearrangePoolBlock(out_chs, group_size)

    def forward(self, x, condition):
        x = self.encode1(x,condition)
        x = self.encode2(x,condition)
        x = self.RearrangePoolBlock(x)
        return x
#thử tạo test case

#DownBlock 
class DownBlock(nn.Module): # ứng dụng cross_ attention vào.
    def __init__(self, in_chs, out_chs, group_size):
        super(DownBlock, self).__init__()
        layers = [
            GELUConvBlock(in_chs, out_chs, group_size), # khả năng ứng dụng nó vào đâyđây
            GELUConvBlock(out_chs, out_chs, group_size),
            RearrangePoolBlock(out_chs, group_size), # thay cho maxpoll inging
        ]
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class UpBlock(nn.Module):
    def __init__(self, in_chs, out_chs, group_size):
        super(UpBlock, self).__init__()
        layers = [
            nn.ConvTranspose2d(2 * in_chs, out_chs, 2, 2),
            GELUConvBlock(out_chs, out_chs, group_size),
            GELUConvBlock(out_chs, out_chs, group_size),
            GELUConvBlock(out_chs, out_chs, group_size),
            GELUConvBlock(out_chs, out_chs, group_size),
        ]
        self.model = nn.Sequential(*layers)

    def forward(self, x, skip):
        x = torch.cat((x, skip), 1)
        x = self.model(x)
        return x


class SinusoidalPositionEmbedBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings


class EmbedBlock(nn.Module):
    def __init__(self, input_dim, emb_dim):
        super(EmbedBlock, self).__init__()
        self.input_dim = input_dim
        layers = [
            nn.Linear(input_dim, emb_dim),
            nn.GELU(),
            nn.Linear(emb_dim, emb_dim),
            nn.Unflatten(1, (emb_dim, 1, 1)),
        ]
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        x = x.view(-1, self.input_dim)
        return self.model(x)


class ResidualConvBlock(nn.Module):
    def __init__(self, in_chs, out_chs, group_size):
        super().__init__()
        self.conv1 = GELUConvBlock(in_chs, out_chs, group_size)
        self.conv2 = GELUConvBlock(out_chs, out_chs, group_size)

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        out = x1 + x2
        return out


class UNet(nn.Module):
    def __init__(
        self, T, img_ch, img_size, down_chs=(64, 64, 128), t_embed_dim=8, c_embed_dim=10
    ):
        super().__init__()
        self.T = T
        up_chs = down_chs[::-1]  # Reverse of the down channels
        latent_image_size = img_size // 4  # 2 ** (len(down_chs) - 1)
        small_group_size = 8
        big_group_size = 32
        dim_embeding = 512

        # Inital convolution
        self.down0 = ResidualConvBlock(img_ch, down_chs[0], small_group_size)

        # Downsample
        # self.down1 = DownBlock(down_chs[0], down_chs[1], big_group_size)
        # self.down2 = DownBlock(down_chs[1], down_chs[2], big_group_size)
        # self.to_vec = nn.Sequential(nn.Flatten(), nn.GELU())
        #remake bằng cách cross-attention. 
        self.down1 = ReDownBlock(down_chs[0], down_chs[1],dim_embeding, big_group_size)
        self.down2 = ReDownBlock(down_chs[1], down_chs[2],dim_embeding, big_group_size)
        self.to_vec = nn.Sequential(nn.Flatten(), nn.GELU())
        
        # Embeddings
        self.dense_emb = nn.Sequential(
            nn.Linear(down_chs[2] * latent_image_size**2, down_chs[1]),
            nn.ReLU(),
            nn.Linear(down_chs[1], down_chs[1]),
            nn.ReLU(),
            nn.Linear(down_chs[1], down_chs[2] * latent_image_size**2),
            nn.ReLU(),
        )
        self.sinusoidaltime = SinusoidalPositionEmbedBlock(t_embed_dim)
        self.t_emb1 = EmbedBlock(t_embed_dim, up_chs[0])
        self.t_emb2 = EmbedBlock(t_embed_dim, up_chs[1])
        self.c_embed1 = EmbedBlock(c_embed_dim, up_chs[0])
        self.c_embed2 = EmbedBlock(c_embed_dim, up_chs[1])

        # Upsample
        self.up0 = nn.Sequential(
            nn.Unflatten(1, (up_chs[0], latent_image_size, latent_image_size)),
            GELUConvBlock(up_chs[0], up_chs[0], big_group_size),
        )
        self.up1 = UpBlock(up_chs[0], up_chs[1], big_group_size)
        self.up2 = UpBlock(up_chs[1], up_chs[2], big_group_size)

        # Match output channels and one last concatenation
        self.out = nn.Sequential(
            nn.Conv2d(2 * up_chs[-1], up_chs[-1], 3, 1, 1),
            nn.GroupNorm(small_group_size, up_chs[-1]),
            nn.ReLU(),
            nn.Conv2d(up_chs[-1], img_ch, 3, 1, 1),
        )

    def forward(self, x, t, c, c1, c_mask):

        down0 = self.down0(x)
        down1 = self.down1(down0,c1)
        down2 = self.down2(down1,c1)
        latent_vec = self.to_vec(down2)

        latent_vec = self.dense_emb(latent_vec)
        t = t.float() / self.T  # Convert from [0, T] to [0, 1]
        t = self.sinusoidaltime(t)
        t_emb1 = self.t_emb1(t)
        t_emb2 = self.t_emb2(t)

        c = c * c_mask
        c_emb1 = self.c_embed1(c)
        c_emb2 = self.c_embed2(c)

        up0 = self.up0(latent_vec)
        up1 = self.up1(c_emb1 * up0 + t_emb1, down2)
        up2 = self.up2(c_emb2 * up1 + t_emb2, down1)
        return self.out(torch.cat((up2, down0), 1))

    
def get_context_mask(c, drop_prob, num_classes):
    c_hot = F.one_hot(c.to(torch.int64), num_classes=num_classes).to(device)
    c_mask = torch.bernoulli(torch.ones_like(c_hot).float() - drop_prob).to(device)
    return c_hot, c_mask


import torch
import torch.nn as nn

def test_cross_attention():
    dim, context_dim, batch_size, seq_len = 32, 64, 2, 16
    cross_attn = CrossAttention(dim, context_dim)
    x = torch.randn(batch_size, seq_len, dim)  # Input sequence
    context = torch.randn(batch_size, seq_len, context_dim)  # Context sequence
    output = cross_attn(x, context)
    assert output.shape == x.shape, f"Expected shape {x.shape}, but got {output.shape}"
    print("CrossAttention test passed!")

def test_unet_encoder_with_cross_attention():
    in_ch, out_ch, cond_dim, batch_size, height, width = 3, 64, 128, 2, 32, 32
    encoder = UNetEncoderWithCrossAttention(in_ch, out_ch, cond_dim)
    x = torch.randn(batch_size, in_ch, height, width)
    condition = torch.randn(batch_size, cond_dim)
    output = encoder(x, condition)
    assert output.shape == (batch_size, out_ch, height, width), f"Unexpected output shape: {output.shape}"
    print("UNetEncoderWithCrossAttention test passed!")

def test_re_down_block():
    in_chs, out_chs, dim_embedding, group_size = 3, 64, 128, 2
    batch_size, height, width = 2, 32, 32
    block = ReDownBlock(in_chs, out_chs, dim_embedding, group_size)
    x = torch.randn(batch_size, in_chs, height, width)
    condition = torch.randn(batch_size, dim_embedding)
    output = block(x, condition)
    assert output.shape[1] == out_chs, f"Unexpected channel output: {output.shape[1]}"
    print("ReDownBlock test passed!")

if __name__ == "__main__":
    test_cross_attention()
    test_unet_encoder_with_cross_attention()
    test_re_down_block()
