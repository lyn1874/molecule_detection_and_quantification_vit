"""
Created on 11:02 at 24/11/2021
@author: bo
"""
import torch
import torch.nn as nn
import torch.nn.init as init
from einops import rearrange, repeat
from einops.layers.torch import Rearrange


def pair(t):
    return t if isinstance(t, tuple) else (t, t)


class PreNorm(nn.Module):
    def __init__(self, input_dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(input_dim)
        self.fn = fn

    def forward(self, x):
        return self.fn(self.norm(x))

    def forward_test(self, x, able_to_forward_test):
        if able_to_forward_test:
            return self.fn.forward_test(self.norm(x))
        else:
            return self.fn(self.norm(x))


class MLPBlock(nn.Module):
    def __init__(self, input_dim, hidden_dim, dropout_rate=0.0, initialisation="DNP"):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(input_dim, hidden_dim),
                                 nn.GELU(),
                                 nn.Dropout(dropout_rate),
                                 nn.Linear(hidden_dim, input_dim),
                                 nn.Dropout(dropout_rate))
        self.initialisation = initialisation
        self._initialize_weights()

    def _initialize_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                if self.initialisation == "kaiming":
                    init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity="relu")
                elif self.initialisation == "xavier":
                    init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    init.normal_(m.bias, std=1e-6)

    def forward(self, x):
        return self.net(x)


class MultiHeadAttention(nn.Module):
    def __init__(self, input_dim, num_heads=8, dim_head=64, dropout_rate=0.0, initialisation="DNP"):
        """Args:
        input_dim:
        num_heads: number of head in the multi-head attention
        dim_head:
        dropout: float
        """
        super().__init__()
        inner_dim = dim_head * num_heads
        project_out = not (num_heads == 1 and dim_head == input_dim)
        self.initialisation = initialisation 
        self.num_heads = num_heads
        self.scale = dim_head ** (-0.5)

        self.attend = nn.Softmax(dim=-1)
        self.to_qkv = nn.Linear(input_dim, inner_dim * 3, bias=False)
        self.to_out = nn.Sequential(nn.Linear(inner_dim, input_dim),
                                    nn.Dropout(dropout_rate)) if project_out else nn.Identity()
        self._initialize_weights()

    def _initialize_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                if self.initialisation == "kaiming":
                    init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity="relu")
                elif self.initialisation == "xavier":
                    init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    init.normal_(m.bias, std=1e-6)

    def forward(self, x):
        """Args:
        x: [batch_size, num_seq, input_dim]
        """
        qkv = self.to_qkv(x).chunk(3, dim=-1)  # [batch_size, seq_len, inner_dim] = [batch_size, num_heads * dim_head]
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.num_heads), qkv)
        # this rearrange function is equivelent to
        # qkv[0].view(-1, seq_len, self.num_heads, self.dim_head).permute(0, 2, 1, 3)
        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale  # [batch_size, num_heads, seq_length, seq_length]
        attn = self.attend(dots)
        out = torch.matmul(attn, v)  # [batch_size, num_heads, seq_length, dim_head]  # attn is like the score
        # that I learned based on the query and keys -> then I will present the values that are associated with higher
        # attention score
        # Query * Keys -> score -> associate the score with the value -> to give the corresponding values
        out = rearrange(out, 'b h n d -> b n (h d)')  # [batch_size, seq_len, num_heads * dim_head]
        return self.to_out(out)

    def forward_test(self, x):
        qkv = self.to_qkv(x).chunk(3, dim=-1)  # [batch_size, seq_len, inner_dim] = [batch_size, num_heads * dim_head]
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.num_heads), qkv)
        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale  # [batch_size, num_heads, seq_length, seq_length]
        attn = self.attend(dots)
        out = torch.matmul(attn, v)  # [batch_size, num_heads, seq_length, dim_head]  # attn is like the score
        # that I learned based on the query and keys -> then I will present the values that are associated with higher
        # attention score
        # Query * Keys -> score -> associate the score with the value -> to give the corresponding values
        out = rearrange(out, 'b h n d -> b n (h d)')  # [batch_size, seq_len, num_heads * dim_head]
        return [q, k, v], attn, self.to_out(out)


class Transformer(nn.Module):
    def __init__(self, input_dim, depth, num_heads, dim_head, mlp_dim, dropout_rate=0.0, initialisation="xavier"):
        super().__init__()
        layers = nn.ModuleList([])
        for i in range(depth):
            layers.append(nn.ModuleList([
                PreNorm(input_dim, MultiHeadAttention(input_dim, num_heads, dim_head, dropout_rate, initialisation)),
                PreNorm(input_dim, MLPBlock(input_dim, mlp_dim, dropout_rate, initialisation))
            ]))
        self.layers = layers

    def forward(self, x):
        for attn, mlp_ff in self.layers:
            x = attn(x) + x
            x = mlp_ff(x) + x
        return x

    def forward_test(self, x):
        attention_maps = []
        qkv_maps = []
        for attn, mlp_ff in self.layers:
            qkv, _attn_map, out = attn.forward_test(x, True)
            x = x + out
            x = mlp_ff(x) + x
            attention_maps.append(_attn_map.detach().cpu().numpy())
            qkv_maps.append([v.detach().cpu().numpy() for v in qkv])
        return qkv_maps, attention_maps, x


class ViT(nn.Module):
    def __init__(self, image_size, patch_size, num_classes, input_dim, depth, num_heads, mlp_dim, pool="cls",
                 channels=3, dim_head=64, dropout=0.0, emb_dropout=0.0, representation_size=0,
                 add_positional_encoding=True, quantification=False, detection=True, 
                 initialisation="xavier"):
        super().__init__()
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)
        print(image_height, image_width, patch_height, patch_width)
        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'image dim % patch dim == 0'

        num_patches = (image_height // patch_height) * (image_width // patch_width)
        patch_dim = channels * patch_height * patch_width

        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'
        self.add_positional_encoding = add_positional_encoding
        print("Patch height", patch_height, "Patch width", patch_width, "Image shape", image_height, image_width)
        if patch_width == 1:
            split_layer = Rearrange('b c (h p1) (w p2) -> b (h w) (p2 p1 c)', p1=patch_height, p2=patch_width)
        else:
            split_layer = Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=patch_height, p2=patch_width)

        self.to_patch_embedding = nn.Sequential(
            split_layer,
            # batch_size, num_patches, patch_height*patch_width*channel
            nn.Linear(patch_dim, input_dim),
        )
        if add_positional_encoding:
            self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, input_dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, input_dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(input_dim, depth, num_heads, dim_head, mlp_dim, dropout, initialisation=initialisation)

        self.pool = pool

        self.layer_norm_after_transformer = nn.LayerNorm(input_dim)
        self.representation_size = representation_size
        self.num_classes = num_classes
        self.quantification = quantification
        self.detection = detection 

        if representation_size != 0:
            self.representation_layer = nn.Linear(input_dim, representation_size)
        else:
            self.representation_layer = nn.Identity()

        if num_classes != 0 and self.detection is True:
            self.mlp_head = nn.Linear(input_dim, num_classes)
        if quantification:
            self.quantification_head = nn.Sequential(nn.Linear(input_dim, 16),
                                                        nn.Linear(16, 1))

    def forward(self, img):
        x = self.to_patch_embedding(img)
        b, n, _ = x.shape

        cls_tokens = repeat(self.cls_token, '() n d -> b n d', b=b)

        x = torch.cat((cls_tokens, x), dim=1)  # yes, because the position 0 gets a learnable class embedding
        if self.add_positional_encoding:
            x += self.pos_embedding[:, :(n+1)]  # why do I need to choose : (n+1) ->

        x = self.dropout(x)

        x = self.transformer(x)
        x = self.layer_norm_after_transformer(x)
        x = x.mean(dim=1) if self.pool == 'mean' else x[:, 0]

        x = self.representation_layer(x)
        # print(x.shape, x.min(), x.max())
        if self.num_classes != 0 and self.detection:
            pred = self.mlp_head(x)
        else:
            pred = []
        if self.quantification:
            quan = self.quantification_head(x).squeeze(1)
        else:
            quan = []
        return x, pred, quan

    def forward_test(self, img):
        x = self.to_patch_embedding(img)
        b, n, _ = x.shape
        cls_tokens = repeat(self.cls_token, '() n d -> b n d', b=b)
        x = torch.cat((cls_tokens, x), dim=1)  # yes, because the position 0 gets a learnable class embedding

        if self.add_positional_encoding is True:
            x += self.pos_embedding[:, :(n + 1)]  # why do I need to choose : (n+1) ->

        x = self.dropout(x)
        query_key_value, attention_maps, x = self.transformer.forward_test(x)

        x = self.layer_norm_after_transformer(x)

        x = x.mean(dim=1) if self.pool == 'mean' else x[:, 0]

        x = self.representation_layer(x)

        if self.num_classes != 0 and self.detection:
            pred = self.mlp_head(x)
        else:
            pred = []

        if self.quantification:
            quan = self.quantification_head(x).squeeze(1)
        else:
            quan = []

        return x, pred, quan, query_key_value, attention_maps








