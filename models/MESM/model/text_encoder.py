from collections import OrderedDict
from typing import Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from tqdm import tqdm

torch.autograd.set_detect_anomaly(True)

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1):
        super().__init__()

        # all conv layers have stride 1. an avgpool is performed after the second convolution when stride > 1
        self.conv1 = nn.Conv2d(inplanes, planes, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)

        self.conv2 = nn.Conv2d(planes, planes, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.avgpool = nn.AvgPool2d(stride) if stride > 1 else nn.Identity()

        self.conv3 = nn.Conv2d(planes, planes * self.expansion, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = None
        self.stride = stride

        if stride > 1 or inplanes != planes * Bottleneck.expansion:
            # downsampling layer is prepended with an avgpool, and the subsequent convolution has stride 1
            self.downsample = nn.Sequential(OrderedDict([
                ("-1", nn.AvgPool2d(stride)),
                ("0", nn.Conv2d(inplanes, planes * self.expansion, 1, stride=1, bias=False)),
                ("1", nn.BatchNorm2d(planes * self.expansion))
            ]))

    def forward(self, x: torch.Tensor):
        identity = x

        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.avgpool(out)
        out = self.bn3(self.conv3(out))

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)
        return out


class AttentionPool2d(nn.Module):
    def __init__(self, spacial_dim: int, embed_dim: int, num_heads: int, output_dim: int = None):
        super().__init__()
        self.positional_embedding = nn.Parameter(torch.randn(spacial_dim ** 2 + 1, embed_dim) / embed_dim ** 0.5)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.c_proj = nn.Linear(embed_dim, output_dim or embed_dim)
        self.num_heads = num_heads

    def forward(self, x):
        x = x.reshape(x.shape[0], x.shape[1], x.shape[2] * x.shape[3]).permute(2, 0, 1)  # NCHW -> (HW)NC
        x = torch.cat([x.mean(dim=0, keepdim=True), x], dim=0)  # (HW+1)NC
        x = x + self.positional_embedding[:, None, :].to(x.dtype)  # (HW+1)NC
        x, _ = F.multi_head_attention_forward(
            query=x, key=x, value=x,
            embed_dim_to_check=x.shape[-1],
            num_heads=self.num_heads,
            q_proj_weight=self.q_proj.weight,
            k_proj_weight=self.k_proj.weight,
            v_proj_weight=self.v_proj.weight,
            in_proj_weight=None,
            in_proj_bias=torch.cat([self.q_proj.bias, self.k_proj.bias, self.v_proj.bias]),
            bias_k=None,
            bias_v=None,
            add_zero_attn=False,
            dropout_p=0,
            out_proj_weight=self.c_proj.weight,
            out_proj_bias=self.c_proj.bias,
            use_separate_proj_weight=True,
            training=self.training,
            need_weights=False
        )

        return x[0]


class ModifiedResNet(nn.Module):
    """
    A ResNet class that is similar to torchvision's but contains the following changes:
    - There are now 3 "stem" convolutions as opposed to 1, with an average pool instead of a max pool.
    - Performs anti-aliasing strided convolutions, where an avgpool is prepended to convolutions with stride > 1
    - The final pooling layer is a QKV attention instead of an average pool
    """

    def __init__(self, layers, output_dim, heads, input_resolution=224, width=64):
        super().__init__()
        self.output_dim = output_dim
        self.input_resolution = input_resolution

        # the 3-layer stem
        self.conv1 = nn.Conv2d(3, width // 2, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(width // 2)
        self.conv2 = nn.Conv2d(width // 2, width // 2, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(width // 2)
        self.conv3 = nn.Conv2d(width // 2, width, kernel_size=3, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(width)
        self.avgpool = nn.AvgPool2d(2)
        self.relu = nn.ReLU(inplace=True)

        # residual layers
        self._inplanes = width  # this is a *mutable* variable used during construction
        self.layer1 = self._make_layer(width, layers[0])
        self.layer2 = self._make_layer(width * 2, layers[1], stride=2)
        self.layer3 = self._make_layer(width * 4, layers[2], stride=2)
        self.layer4 = self._make_layer(width * 8, layers[3], stride=2)

        embed_dim = width * 32  # the ResNet feature dimension
        self.attnpool = AttentionPool2d(input_resolution // 32, embed_dim, heads, output_dim)

    def _make_layer(self, planes, blocks, stride=1):
        layers = [Bottleneck(self._inplanes, planes, stride)]

        self._inplanes = planes * Bottleneck.expansion
        for _ in range(1, blocks):
            layers.append(Bottleneck(self._inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        def stem(x):
            for conv, bn in [(self.conv1, self.bn1), (self.conv2, self.bn2), (self.conv3, self.bn3)]:
                x = self.relu(bn(conv(x)))
            x = self.avgpool(x)
            return x

        x = x.type(self.conv1.weight.dtype)
        x = stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.attnpool(x)

        return x


class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)


class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)


class ResidualAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None):
        super().__init__()

        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.ln_2 = LayerNorm(d_model)
        self.attn_mask = attn_mask

    def attention(self, x: torch.Tensor):
        self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
        return self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]

    def forward(self, x: torch.Tensor):
        x = x + self.attention(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class Transformer(nn.Module):
    def __init__(self, width: int, layers: int, heads: int, attn_mask: torch.Tensor = None):
        super().__init__()
        self.width = width
        self.layers = layers
        self.resblocks = nn.Sequential(*[ResidualAttentionBlock(width, heads, attn_mask) for _ in range(layers)])

    def forward(self, x: torch.Tensor):
        return self.resblocks(x)


class VisualTransformer(nn.Module):
    def __init__(self, input_resolution: int, patch_size: int, width: int, layers: int, heads: int, output_dim: int):
        super().__init__()
        self.input_resolution = input_resolution
        self.output_dim = output_dim
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=width, kernel_size=patch_size, stride=patch_size, bias=False)

        scale = width ** -0.5
        self.class_embedding = nn.Parameter(scale * torch.randn(width))
        self.positional_embedding = nn.Parameter(scale * torch.randn((input_resolution // patch_size) ** 2 + 1, width))
        self.ln_pre = LayerNorm(width)

        self.transformer = Transformer(width, layers, heads)

        self.ln_post = LayerNorm(width)
        self.proj = nn.Parameter(scale * torch.randn(width, output_dim))

    def forward(self, x: torch.Tensor):
        x = self.conv1(x)  # shape = [*, width, grid, grid]
        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
        x = torch.cat([self.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x], dim=1)  # shape = [*, grid ** 2 + 1, width]
        x = x + self.positional_embedding.to(x.dtype)
        x = self.ln_pre(x)

        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD

        x = self.ln_post(x[:, 0, :])

        if self.proj is not None:
            x = x @ self.proj

        return x


class CLIPTextEncoder(nn.Module):
    def __init__(self,
                 embed_dim: int,
                 # vision
                #  image_resolution: int,
                #  vision_layers: Union[Tuple[int, int, int, int], int],
                #  vision_width: int,
                #  vision_patch_size: int,
                 # text
                 context_length: int,
                 vocab_size: int,
                 transformer_width: int,
                 transformer_heads: int,
                 transformer_layers: int
                 ):
        super().__init__()

        self.context_length = context_length

        # if isinstance(vision_layers, (tuple, list)):
        #     vision_heads = vision_width * 32 // 64
        #     self.visual = ModifiedResNet(
        #         layers=vision_layers,
        #         output_dim=embed_dim,
        #         heads=vision_heads,
        #         input_resolution=image_resolution,
        #         width=vision_width
        #     )
        # else:
        #     vision_heads = vision_width // 64
        #     self.visual = VisualTransformer(
        #         input_resolution=image_resolution,
        #         patch_size=vision_patch_size,
        #         width=vision_width,
        #         layers=vision_layers,
        #         heads=vision_heads,
        #         output_dim=embed_dim
        #     )

        self.transformer = Transformer(
            width=transformer_width,
            layers=transformer_layers,
            heads=transformer_heads,
            attn_mask=self.build_attention_mask()
        )

        self.vocab_size = vocab_size
        self.token_embedding = nn.Embedding(vocab_size, transformer_width)
        self.positional_embedding = nn.Parameter(torch.empty(self.context_length, transformer_width))
        self.ln_final = LayerNorm(transformer_width)

        self.text_projection = nn.Parameter(torch.empty(transformer_width, embed_dim))
        # self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

        self.initialize_parameters()

    def initialize_parameters(self):
        nn.init.normal_(self.token_embedding.weight, std=0.02)
        nn.init.normal_(self.positional_embedding, std=0.01)

        # if isinstance(self.visual, ModifiedResNet):
        #     if self.visual.attnpool is not None:
        #         std = self.visual.attnpool.c_proj.in_features ** -0.5
        #         nn.init.normal_(self.visual.attnpool.q_proj.weight, std=std)
        #         nn.init.normal_(self.visual.attnpool.k_proj.weight, std=std)
        #         nn.init.normal_(self.visual.attnpool.v_proj.weight, std=std)
        #         nn.init.normal_(self.visual.attnpool.c_proj.weight, std=std)

        #     for resnet_block in [self.visual.layer1, self.visual.layer2, self.visual.layer3, self.visual.layer4]:
        #         for name, param in resnet_block.named_parameters():
        #             if name.endswith("bn3.weight"):
        #                 nn.init.zeros_(param)

        proj_std = (self.transformer.width ** -0.5) * ((2 * self.transformer.layers) ** -0.5)
        attn_std = self.transformer.width ** -0.5
        fc_std = (2 * self.transformer.width) ** -0.5
        for block in self.transformer.resblocks:
            nn.init.normal_(block.attn.in_proj_weight, std=attn_std)
            nn.init.normal_(block.attn.out_proj.weight, std=proj_std)
            nn.init.normal_(block.mlp.c_fc.weight, std=fc_std)
            nn.init.normal_(block.mlp.c_proj.weight, std=proj_std)

        if self.text_projection is not None:
            nn.init.normal_(self.text_projection, std=self.transformer.width ** -0.5)

    def build_attention_mask(self):
        # lazily create causal attention mask, with full attention between the vision tokens
        # pytorch uses additive attention mask; fill with -inf
        mask = torch.empty(self.context_length, self.context_length)
        mask.fill_(float("-inf"))
        mask.triu_(1)  # zero out the lower diagonal
        return mask

    @property
    def dtype(self):
        return torch.float16

    # def encode_image(self, image):
    #     return self.visual(image.type(self.dtype))

    @torch.no_grad()
    def forward(self, text):
        x = self.token_embedding(text).type(self.dtype)  # [batch_size, n_ctx, d_model]

        x = x + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        eos_x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ self.text_projection

        return dict(last_hidden_state=x, pooler_output=eos_x)

    # def forward(self, image, text):
    #     # image_features = self.encode_image(image)
    #     text_features = self.encode_text(text)

    #     # normalized features
    #     # image_features = image_features / image_features.norm(dim=-1, keepdim=True)
    #     text_features = text_features / text_features.norm(dim=-1, keepdim=True)

    #     # cosine similarity as logits
    #     logit_scale = self.logit_scale.exp()
    #     logits_per_image = logit_scale * image_features @ text_features.t()
    #     logits_per_text = logit_scale * text_features @ image_features.t()

    #     # shape = [global_batch_size, global_batch_size]
    #     return logits_per_image, logits_per_text


def convert_weights(model: nn.Module):
    """Convert applicable model parameters to fp16"""

    def _convert_weights_to_fp16(l):
        if isinstance(l, (nn.Conv1d, nn.Conv2d, nn.Linear)):
            l.weight.data = l.weight.data.half()
            if l.bias is not None:
                l.bias.data = l.bias.data.half()

        if isinstance(l, nn.MultiheadAttention):
            for attr in [*[f"{s}_proj_weight" for s in ["in", "q", "k", "v"]], "in_proj_bias", "bias_k", "bias_v"]:
                tensor = getattr(l, attr)
                if tensor is not None:
                    tensor.data = tensor.data.half()

        for name in ["text_projection", "proj"]:
            if hasattr(l, name):
                attr = getattr(l, name)
                if attr is not None:
                    attr.data = attr.data.half()

    model.apply(_convert_weights_to_fp16)


class GloVe(object):
    """
    Attributes:
        self.glove: {str: tensor}
    """
    def __init__(self, glove_path):
        self.glove_path = glove_path
        self.dim = 300
        self.glove = self._load()
        self.glove["<PAD>"] = torch.zeros(self.dim)
        self.glove["<UNK>"] = torch.randn(self.dim)

    def get(self, word):
        if self.contains(word):
            return self.glove[word]
        else:
            return self.glove["<UNK>"]

    def contains(self, word):
        return word in self.glove.keys()

    def _load(self):
        """ Load GloVe embeddings of this vocabulary.
        """
        glove = dict()
        with open(self.glove_path, 'r') as f:
            for line in tqdm(f.readlines(), desc="Reading GloVe from {}".format(self.glove_path)):
                split_line = line.split()
                word = " ".join(split_line[0: len(split_line) - self.dim])  # some words include space
                embedding = torch.from_numpy(np.array(split_line[-self.dim:], dtype=np.float32))
                glove[word] = embedding

        return glove


class GloveTextEncoder(nn.Module):
    def __init__(self, vocab, glove):
        super(GloveTextEncoder, self).__init__()
        dim = glove.dim
        self.emb = nn.Embedding(
            num_embeddings=len(vocab),
            embedding_dim=dim
        )
        # freeze the GloVe embedding
        for param in self.emb.parameters():
            param.requires_grad = False

        for w in vocab.wtoi.keys():
            self.emb.weight.data[vocab.wtoi[w], :] = glove.get(w)

    def forward(self, word_ids):
        """ Get embedding from word ids, and map the embedding to out_dim.
        Args:
            word_ids: (B, L)
        Returns:
            (B, L, out_dim)
        """
        return self.emb(word_ids)
    
    
class PhraseLevelEncoder(nn.Module):
    def __init__(self, h_dim):
        super(PhraseLevelEncoder, self).__init__()

        self.att_score_l = nn.Linear(self.h_dim, 1)
        self.att_score_r = nn.Linear(self.h_dim, 1)
        
        self.lin_l = nn.Linear(self.h_dim, self.h_dim)
        self.lin_r = nn.Linear(self.h_dim, self.h_dim)

        self.lin_rep = nn.Linear(self.h_dim, self.h_dim)
        self.relu = nn.ReLU(inplace=False)

        self.h_dim = h_dim
    
    def forward(self, h_left, h_right):
        """ Encode the phrase-level feature h_ij by combining h_ik and h_kj.
        Args:
            h_left: a phrase-level feature corresponding to h_ik [batch, h_dim]
            h_right: a phrase-level feature corresponding to h_kj [batch, h_dim]
        Returns:
            h_ij: the phrase-level feature of h_ik and h_kj combined.
        """

        self_weights = torch.softmax(torch.cat((self.att_score_l(h_left), self.att_score_r(h_right)), dim=-1), dim=-1) # [batch, 2]

        values = torch.stack((
            self.lin_l(h_left),
            self.lin_r(h_right)
        ), dim=1) # [batch, h_dim, 2]

        h_hat = torch.matmul(values, self_weights.unsqueeze(1)).squeeze(1) #! Double Check
        # self_weights.matmul(torch.transpose(values, -1, -2)) # [batch, h_dim]
        
        output = self.relu(self.lin_rep(h_hat)) + h_hat # [batch, h_dim]

        return output
    
        
class GLT(nn.Module):
    def __init__(self, args):
        super(GLT, self).__init__()

        self.h_dim = args.t_feat_dim

        self.att_score_l = nn.Linear(self.h_dim, 1)
        self.att_score_r = nn.Linear(self.h_dim, 1)
        
        self.lin_l = nn.Linear(self.h_dim, self.h_dim)
        self.lin_r = nn.Linear(self.h_dim, self.h_dim)

        self.lin_rep = nn.Linear(self.h_dim, self.h_dim)
        self.relu = nn.ReLU(inplace=False)

        self.weight_split = nn.Linear(self.h_dim, 1) # used to compute `a = softmax(s^T * h^k)`
    
    def forward(self, words_feats, words_masks):
        """ computes the phrase-level features using the CYK algorithm.
            This computation is not "batched".
        Args:
            words_feat: word embeddings extracted from a pre-trained model --> [batch, sentence length, h_dim]
            words_mask: binary mask that denotes each sentence length in a batched sample --> [batch, sentence length, h_dim]

        Returns:
            sentence_feat: sentence-level feature --> [batch, h_dim]

        Example:
            feature_table:
            +-----+-----+-----+-----+-----+
            | w_1 | w_2 | w_3 | w_4 | w_5 | ---> word-level features
            +-----+-----+-----+-----+-----+
            | h_12| h_23| h_34| h_45|     | -+
            +-----+-----+-----+-----+-----+  |
            | h_13| h_24| h_35|     |     |  |-> phrase-level features
            +-----+-----+-----+-----+-----+  | 
            | h_14| h_25|     |     |     | -+ 
            +-----+-----+-----+-----+-----+ -+
            | h_15|     |     |     |     |  |-> sentence feature
            +-----+-----+-----+-----+-----+ -+
        
        """

        batch, masked_len, _h_dim = words_feats.shape
        unmasked_len = torch.sum(words_masks, dim=-1).squeeze(-1).int()

        # print(f"{words_feats.shape}")
        # print()
        # print(f"{unmasked_len}")
        # print()
        
        sentence_feats = torch.zeros((batch, self.h_dim)).to("cuda")
        feature_tables = torch.zeros((batch, masked_len, masked_len, self.h_dim)).to("cuda")
        split_distribution = torch.zeros((batch, masked_len, masked_len, masked_len)).to("cuda")

        attn_weights = torch.zeros((batch, masked_len, masked_len, masked_len, 2)).to("cuda")
        temp_h = torch.zeros((batch, masked_len, masked_len, masked_len, self.h_dim)).to("cuda")
        h_hat = torch.zeros((batch, masked_len, masked_len, masked_len, self.h_dim)).to("cuda")

        for b in range(batch):
            i = 1 # height
            length = unmasked_len[b].item()
            # print(f"length: {length}")

            for _ in range(length):
                feature_tables[b, 0, _] = words_feats[b, _]

            while (i < length):
                j = 0 # width
                while (i + j < length):
                    for k in range(i):
                        """
                        Combine feature_tables[b, k, j] and feature_tables[b, i-k-1, j+k+1]
                        """
                        # print(f"({b}, {i}, {j}, {k})")
                        attn_weights[b, i, j, k] = torch.softmax(torch.cat((self.att_score_l(feature_tables[b, k, j]), self.att_score_r(feature_tables[b, i-k-1, j+k+1])), dim=-1), dim=-1)

                        h_hat[b, i, j, k] = torch.matmul(attn_weights[b,i, j, k], torch.stack((feature_tables[b, k, j], feature_tables[b, i-k-1, j+k+1])))
 
                        temp_h[b, i, j, k] = self.relu(self.lin_rep(h_hat[b, i, j, k])) + h_hat[b, i, j, k] 
                    
                    split_distribution[b, i, j] = torch.softmax(self.weight_split(temp_h[b, i, j]), dim=0).squeeze()

                    feature_tables[b, i, j] = torch.matmul(temp_h[b, i, j].T, split_distribution[b, i, j]).squeeze()

                    j += 1
                i += 1
            sentence_feats[b] = feature_tables[b, i-1, j-1, :]
        
        return sentence_feats

# class GLT(nn.Module):
#     def __init__(self):
#         super(GLT, self).__init__()

#         self.W_q = nn.Linear(300, 100)
#         self.W_k = nn.Linear(300, 100)
    
#     def forward(self, words_feats):
#         """
#         args:
#             words_feats: [batch, masked_len, 300]
#         """
#         avg_sent_feat = torch.sum(words_feats, dim=-2) # Avgeraged Sentence Feature: [batch, 300]

#         query = self.W_q(avg_sent_feat) # [batch, 100]
#         key = self.W_k(words_feats)     # [batch, masked_len, 100]

#         query_expanded = query.unsqueeze(1)                        # [batch, 1, 100]
#         dot_product = torch.sum(query_expanded * key, dim=-1)      # [batch, masked_len]
#         self.attn_weights = F.softmax(dot_product, dim=-1)         # [batch, masked_len]

#         attn_expanded = self.attn_weights.unsqueeze(-1) # [batch, masked_len, 1]
#         sent_feats = words_feats * attn_expanded       # [batch, masked_len, 300]
#         sent_feats = sent_feats.sum(dim=-2)           # [batch, 300]

#         return sent_feats


