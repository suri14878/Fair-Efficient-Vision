# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# Partly revised by YZ @UCL&Moorfields
# --------------------------------------------------------

from functools import partial

import torch
import torch.nn as nn

import timm.models.vision_transformer
from pos_embed import interpolate_pos_embed
from timm.models.layers import trunc_normal_
import timm

class VisionTransformer(timm.models.vision_transformer.VisionTransformer):
    """ Vision Transformer with support for global average pooling
    """
    def __init__(self, global_pool=False, **kwargs):
        super(VisionTransformer, self).__init__(**kwargs)
        print("Global Pooling Current State : " + str(global_pool))
        self.global_pool = global_pool
        self.head_drop = nn.Dropout(0)
        print(self.num_classes)
        self.head = nn.Linear(kwargs['embed_dim'], self.num_classes) if self.num_classes > 0 else nn.Identity()
        self.project_layer = nn.Linear(1024, 768)
        if self.global_pool:
            norm_layer = kwargs['norm_layer']
            embed_dim = kwargs['embed_dim']
            self.fc_norm = norm_layer(embed_dim)
            print(self.norm)
            del self.norm 

    def forward_features(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)
        
        # Ensure patch_embed returns only the tensor
        if isinstance(x, tuple):
            print('tuple')
            x, _ = x  # Unpack if it's a tuple
        
        cls_tokens = self.cls_token.expand(B, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)

        for blk in self.blocks:
            x = blk(x)
        
        if self.global_pool:
#             print(x)
            x = x[:, 1:, :].mean(dim=1)  # global pool without cls token
#             print('Global Pool without CLS Token')
            outcome = self.fc_norm(x)
        else:
            x = self.norm(x)
            outcome = x[:, 0]

        return outcome
    
    
    def get_spatial_feature_maps(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)  

        if isinstance(x, tuple):
            x = x[0]

        # Exclude CLS token and reshape into spatial grid
        num_patches = int((x.size(1))**0.5)  # Assuming square grid
        spatial_features = x.permute(0, 2, 1).reshape(B, -1, num_patches, num_patches)  # [B, embed_dim, H, W]

        return spatial_features

    def forward(self, x) -> torch.Tensor:
        x = self.forward_features(x)
        logit = x.clone()
        if self.embed_dim > 768:
            logit = self.project_layer(logit)
        x = self.fc_norm(x)
        x = self.head_drop(x)
        x=self.head(x)
        return x,logit

def vit_large_patch16(**kwargs):
    model = VisionTransformer(
        patch_size=16, embed_dim=1024, depth=24, num_heads=16, mlp_ratio=4, qkv_bias=True,#num_classes = 3,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

def load_pretrained_vit_base(global_pool=False, num_classes=3, target_size=224):
    pretrained_model = timm.create_model('vit_base_patch16_224', pretrained=True)
    checkpoint_model = pretrained_model.state_dict()
    
    model = VisionTransformer(
        patch_size=16, embed_dim =768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), global_pool=global_pool, img_size=target_size
    )
    
    interpolate_pos_embed(model, checkpoint_model)
    
    model.load_state_dict(checkpoint_model, strict=False)
    
    model.head = nn.Linear(model.embed_dim, num_classes)
    trunc_normal_(model.head.weight, std=2e-5)
    if model.head.bias is not None:
        nn.init.constant_(model.head.bias, 0)

    return model

