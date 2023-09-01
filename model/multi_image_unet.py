import torch
import torch.nn as nn

from .unet_parts import (
    ResnetBlock, 
    ResnetBlock2D,
    SinusoidalPosEmb, 
    PreNorm,
    PreNorm2D,
    LinearAttention,
    LinearAttention2D,
    DecoderCrossAttention,
    Residual,
    Upsample,
    Upsample2D,
    Downsample,
    Downsample2D
)
# adapted from https://github.com/lucidrains/denoising-diffusion-pytorch/blob/main/denoising_diffusion_pytorch/denoising_diffusion_pytorch.py#L97

class DDPMUNet(nn.Module):
    """
    3D DDPM U-Net which accepts input shaped as
    B x Cond x Channels x Height x Width x Depth
    """
    def __init__(self, cfg):
        super(DDPMUNet, self).__init__()
        
        self.cfg = cfg

        # input dimensions and initial convolutional layer
        in_channels = self.cfg.model.unet.input_dim
        if self.cfg.model.unet.self_condition and \
                not self.cfg.model.feature_extractor_2d.use:
            in_channels += 3
        dim = self.cfg.model.unet.model_channels
        self.init_conv = nn.Conv3d(in_channels, dim, 7, padding = 3)

        # ========== time embedding ==========
        time_dim = dim * 4
        sinu_pos_emb = SinusoidalPosEmb(dim)
        fourier_dim = dim
        self.time_mlp = nn.Sequential(
            sinu_pos_emb,
            nn.Linear(fourier_dim, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim)
        )

        # ========== unet channels ==========
        channel_mult = self.cfg.model.unet.channel_mult
        self.attn_resolutions = self.cfg.model.unet.attn_resolutions
        dims = [dim, *map(lambda m: dim * m, channel_mult)]
        in_out = list(zip(dims[:-1], dims[1:]))
        # channels dimensions of intermediate feature maps
        self.ft_chans = []
        # spatial dimensions of intermediate feature maps
        current_side = cfg.model.volume_size
        self.sides = []

        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])
        if self.cfg.model.unet.attention_aggregation:
            self.volume_aggregators = []
        num_resolutions = len(in_out)

        # ========== unet layers ==========
        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)
            layers = []
            # resnet blocks
            for b_idx in range(cfg.model.unet.blocks_per_res):
                layers.append(ResnetBlock(dim_in, dim_in, time_emb_dim = time_dim))
                self.ft_chans.append(dim_in)
                self.sides.append(current_side)
            # attention
            if current_side in self.attn_resolutions:
                layers.append(Residual(PreNorm(dim_in, LinearAttention(dim_in))))
            # downsampling
            layers.append(Downsample(dim_in, dim_out) if not is_last else nn.Conv3d(dim_in, dim_out, 3, padding = 1))
            current_side = current_side // 2 if not is_last else current_side
            self.downs.append(nn.ModuleList([*layers]))

        mid_dim = dims[-1]
        self.mid_block1 = ResnetBlock(mid_dim, mid_dim, time_emb_dim = time_dim)
        self.mid_attn = Residual(PreNorm(mid_dim, LinearAttention(mid_dim)))
        self.mid_block2 = ResnetBlock(mid_dim, mid_dim, time_emb_dim = time_dim)
        self.ft_chans.append(mid_dim)
        self.sides.append(current_side)

        if self.cfg.model.unet.attention_aggregation:
            self.volume_aggregators.append(DecoderCrossAttention(mid_dim, mid_dim, cfg.model.n_heads,
                                                                 include_query_as_key = False))
            self.query_volume = nn.Parameter(data = torch.rand((mid_dim,
                                                                current_side,
                                                                current_side,
                                                                current_side)),
                                                                requires_grad=True)

        for ind, (dim_in, dim_out) in enumerate(reversed(in_out)):
            is_last = ind == (len(in_out) - 1)

            layers = []
            for b_idx in range(cfg.model.unet.blocks_per_res):
                if self.cfg.model.unet.attention_aggregation:
                    self.volume_aggregators.append(DecoderCrossAttention(dim_in, dim_out, cfg.model.n_heads,
                                                                         include_query_as_key = False))

                layers.append(ResnetBlock(dim_out + dim_in, dim_out, time_emb_dim = time_dim))
            layers.append(Residual(PreNorm(dim_out, LinearAttention(dim_out))))
            layers.append(Upsample(dim_out, dim_in) if not is_last else nn.Conv3d(dim_out, dim_in, 3, padding = 1))
            self.ups.append(nn.ModuleList([*layers]))

        if self.cfg.model.unet.attention_aggregation:
            self.volume_aggregators.append(DecoderCrossAttention(dim, dim, cfg.model.n_heads,
                                                                 include_query_as_key = False))
            self.volume_aggregators = nn.ModuleList(self.volume_aggregators[::-1])

        self.final_res_block = ResnetBlock(dim * 2, dim, time_emb_dim = time_dim)

        # ========== 3D conv upsampler =========
        if self.cfg.model.volume_size != self.cfg.render.volume_size:
            self.conv_upsampler = nn.Sequential(
                nn.Upsample(scale_factor = 2, mode = 'nearest'),
                ResnetBlock(dim, dim),
                ResnetBlock(dim, dim)
            )
        else:
            self.conv_upsampler = nn.Identity()

        # ========== output layers ==========
        if self.cfg.model.explicit_volume:
            self.out_color = nn.Conv3d(dim, 3, 1)
            self.out_sigma = nn.Conv3d(dim, 1, 1)

            # initialization of the output layer - see SingleLayerReconstructor for explanation
            fc_gain = (cfg.render.max_depth-cfg.render.min_depth) / cfg.render.n_pts_per_ray
            nn.init.xavier_uniform_(self.out_sigma.weight, fc_gain)
            nn.init.constant_(self.out_sigma.bias, 4.0 * fc_gain)

    def forward(self, x, t):
        """
        volumes: (B x Cond x C x D x H x W)
        t: (B x Cond)
        """
        B, Cond, C, D, H, W = x.shape

        encoder_emb = self.time_mlp(t.reshape(B*Cond,))
        # need some options here, for now maxpool the embedding across the set
        decoder_emb = encoder_emb.reshape(B, Cond, -1).max(dim=1, keepdim=False)[0]

        ft_map_idx = 0

        x = self.init_conv(x.reshape(-1, C, H, W, D))
        r = x.reshape(B, Cond, self.ft_chans[ft_map_idx], 
                      self.sides[ft_map_idx],
                      self.sides[ft_map_idx],
                      self.sides[ft_map_idx]).clone()

        h = []
        for down in self.downs:
            res_blocks = down[:self.cfg.model.unet.blocks_per_res]
            if self.sides[ft_map_idx] in self.attn_resolutions:
                attn, downsample = down[self.cfg.model.unet.blocks_per_res:]
            else:
                downsample = down[self.cfg.model.unet.blocks_per_res:][0]
            for r_idx, res_block in enumerate(res_blocks):
                x = res_block(x, encoder_emb)
                if r_idx == self.cfg.model.unet.blocks_per_res - 1 \
                        and self.sides[ft_map_idx] in self.attn_resolutions:
                    x = attn(x)
                h.append(x.reshape(B, Cond, self.ft_chans[ft_map_idx], 
                                                       self.sides[ft_map_idx], 
                                                       self.sides[ft_map_idx], 
                                                       self.sides[ft_map_idx]))
                ft_map_idx += 1
            x = downsample(x)

        x = self.mid_block1(x, encoder_emb)
        x = self.mid_attn(x)
        x = self.mid_block2(x, encoder_emb)

        x = x.reshape(B, Cond, self.ft_chans[ft_map_idx], 
                      self.sides[ft_map_idx], 
                      self.sides[ft_map_idx], 
                      self.sides[ft_map_idx])

        ft_map_idx += 1
        if self.cfg.model.unet.attention_aggregation:
            x = self.volume_aggregators[ft_map_idx](x, self.query_volume)
        else:
            x = torch.mean(x, dim=1, keepdim=False)
        ft_map_idx -= 1

        for up in self.ups:
            res_blocks = up[:self.cfg.model.unet.blocks_per_res]
            attn, upsample = up[self.cfg.model.unet.blocks_per_res:]

            for r_idx, res_block in enumerate(res_blocks):
                scf = h.pop()
                if self.cfg.model.unet.attention_aggregation:
                    scf = self.volume_aggregators[ft_map_idx](scf, x)
                else:
                    scf = torch.mean(scf, dim=1, keepdim=False)
                ft_map_idx -= 1
                x = torch.cat((x, scf), dim = 1)
                x = res_block(x, decoder_emb)

            x = attn(x)
            x = upsample(x)

        assert ft_map_idx == 0
        if self.cfg.model.unet.attention_aggregation:
            x = torch.cat((x, self.volume_aggregators[ft_map_idx](r, x)), dim = 1)
        else:
            x = torch.cat((x, torch.mean(r, dim=1, keepdim=False)), dim = 1)

        x = self.final_res_block(x, decoder_emb)

        x = self.conv_upsampler(x)

        if self.cfg.model.explicit_volume:
            colors = self.out_color(x)
            sigma = self.out_sigma(x)
        else:
            colors = x
            sigma = torch.empty_like(x[:, :1, ...], device=x.device)

        return sigma, colors

class TriplaneUNet(nn.Module):
    """
    Triplane U-Net is a 2D U-Net.
    """
    def __init__(self, cfg):
        super(TriplaneUNet, self).__init__()
        
        self.cfg = cfg

        # input dimensions and initial convolutional layer
        in_channels = self.cfg.model.unet.input_dim
        if self.cfg.model.unet.self_condition and \
                not self.cfg.model.feature_extractor_2d.use:
            in_channels += 3
        dim = self.cfg.model.unet.model_channels
        self.init_conv = nn.Conv2d(in_channels, dim, 7, padding = 3)

        # ========== time embedding ==========
        time_dim = dim * 4
        sinu_pos_emb = SinusoidalPosEmb(dim)
        fourier_dim = dim
        self.time_mlp = nn.Sequential(
            sinu_pos_emb,
            nn.Linear(fourier_dim, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim)
        )

        # ========== unet channels ==========
        channel_mult = self.cfg.model.unet.channel_mult
        self.attn_resolutions = self.cfg.model.unet.attn_resolutions
        dims = [dim, *map(lambda m: dim * m, channel_mult)]
        in_out = list(zip(dims[:-1], dims[1:]))
        # channels dimensions of intermediate feature maps
        self.ft_chans = []
        # spatial dimensions of intermediate feature maps
        current_side = cfg.data.input_size[0]
        self.sides = []

        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])
        if self.cfg.model.unet.attention_aggregation:
            self.volume_aggregators = []
        num_resolutions = len(in_out)

        # ========== unet layers ==========
        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)
            layers = []
            # resnet blocks
            for b_idx in range(cfg.model.unet.blocks_per_res):
                layers.append(ResnetBlock2D(dim_in, dim_in, time_emb_dim = time_dim))
                self.ft_chans.append(dim_in)
                self.sides.append(current_side)
            # attention
            if current_side in self.attn_resolutions:
                layers.append(Residual(PreNorm2D(dim_in, LinearAttention2D(dim_in))))
            # downsampling
            layers.append(Downsample2D(dim_in, dim_out) if not is_last else nn.Conv2d(dim_in, dim_out, 3, padding = 1))
            current_side = current_side // 2 if not is_last else current_side
            self.downs.append(nn.ModuleList([*layers]))

        mid_dim = dims[-1]
        self.mid_block1 = ResnetBlock2D(mid_dim, mid_dim, time_emb_dim = time_dim)
        self.mid_attn = Residual(PreNorm2D(mid_dim, LinearAttention2D(mid_dim)))
        self.mid_block2 = ResnetBlock2D(mid_dim, mid_dim, time_emb_dim = time_dim)
        self.ft_chans.append(mid_dim)
        self.sides.append(current_side)

        for ind, (dim_in, dim_out) in enumerate(reversed(in_out)):
            is_last = ind == (len(in_out) - 1)

            layers = []
            for b_idx in range(cfg.model.unet.blocks_per_res):
                layers.append(ResnetBlock2D(dim_out + dim_in, dim_out, time_emb_dim = time_dim))
            layers.append(Residual(PreNorm2D(dim_out, LinearAttention2D(dim_out))))
            layers.append(Upsample2D(dim_out, dim_in) if not is_last else nn.Conv2d(dim_out, dim_in, 3, padding = 1))
            self.ups.append(nn.ModuleList([*layers]))

        self.final_res_block = ResnetBlock2D(dim * 2, dim * 3, time_emb_dim = time_dim)

    def forward(self, x, t):
        """
        volumes: (B x Cond x C x D x H x W)
        t: (B x Cond)
        """
        B, Cond, C, H, W = x.shape

        encoder_emb = self.time_mlp(t.reshape(B*Cond,))
        # need some options here, for now maxpool the embedding across the set
        decoder_emb = encoder_emb.reshape(B, Cond, -1).max(dim=1, keepdim=False)[0]

        ft_map_idx = 0

        assert x.shape[1] == 1, "Accepting only single-image viewset"
        x = self.init_conv(x.reshape(-1, C, H, W))
        r = x.clone()

        h = []
        for down in self.downs:
            res_blocks = down[:self.cfg.model.unet.blocks_per_res]
            if self.sides[ft_map_idx] in self.attn_resolutions:
                attn, downsample = down[self.cfg.model.unet.blocks_per_res:]
            else:
                downsample = down[self.cfg.model.unet.blocks_per_res:][0]
            for r_idx, res_block in enumerate(res_blocks):
                x = res_block(x, encoder_emb)
                if r_idx == self.cfg.model.unet.blocks_per_res - 1 \
                        and self.sides[ft_map_idx] in self.attn_resolutions:
                    x = attn(x)
                h.append(x)
                ft_map_idx += 1
            x = downsample(x)

        x = self.mid_block1(x, encoder_emb)
        x = self.mid_attn(x)
        x = self.mid_block2(x, encoder_emb)

        for up in self.ups:
            res_blocks = up[:self.cfg.model.unet.blocks_per_res]
            attn, upsample = up[self.cfg.model.unet.blocks_per_res:]

            for r_idx, res_block in enumerate(res_blocks):
                scf = h.pop()
                ft_map_idx -= 1
                x = torch.cat((x, scf), dim = 1)
                x = res_block(x, decoder_emb)

            x = attn(x)
            x = upsample(x)

        assert ft_map_idx == 0
        x = torch.cat((x, r), dim = 1)

        x = self.final_res_block(x, decoder_emb)

        # reshape into triplanes
        return x.reshape(x.shape[0], 3,
                         self.cfg.model.unet.model_channels, 
                         *x.shape[2:])