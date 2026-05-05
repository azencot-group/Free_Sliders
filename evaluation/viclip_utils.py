
import gzip
import html
import ftfy
import regex as re
import functools
import logging
import os
import torch
import torch.nn.functional as F
from pkg_resources import packaging
from torch import nn
from collections import OrderedDict
from einops import rearrange
from timm.layers import DropPath
from timm.models import register_model
import torch.utils.checkpoint as checkpoint
from torchvision.transforms import (
    CenterCrop,
    Compose,
    InterpolationMode,
    Normalize,
    Resize,
    ToTensor,
)

logger = logging.getLogger(__name__)

def bytes_to_unicode():
    """
    Returns list of utf-8 byte and a corresponding list of unicode strings.
    The reversible bpe codes work on unicode strings.
    This means you need a large # of unicode characters in your vocab if you want to avoid UNKs.
    When you're at something like a 10B token dataset you end up needing around 5K for decent coverage.
    This is a signficant percentage of your normal, say, 32K bpe vocab.
    To avoid that, we want lookup tables between utf-8 bytes and unicode strings.
    And avoids mapping to whitespace/control characters the bpe code barfs on.
    """
    bs = list(range(ord("!"), ord("~")+1))+list(range(ord("¡"), ord("¬")+1))+list(range(ord("®"), ord("ÿ")+1))
    cs = bs[:]
    n = 0
    for b in range(2**8):
        if b not in bs:
            bs.append(b)
            cs.append(2**8+n)
            n += 1
    cs = [chr(n) for n in cs]
    return dict(zip(bs, cs))

def get_pairs(word):
    """Return set of symbol pairs in a word.
    Word is represented as tuple of symbols (symbols being variable-length strings).
    """
    pairs = set()
    prev_char = word[0]
    for char in word[1:]:
        pairs.add((prev_char, char))
        prev_char = char
    return pairs

def basic_clean(text):
    text = ftfy.fix_text(text)
    text = html.unescape(html.unescape(text))
    return text.strip()

def whitespace_clean(text):
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
    return text

def clip_image_transform(n_px=224):
    return Compose([
        Resize(n_px, interpolation=InterpolationMode.BICUBIC),
        CenterCrop(n_px),
        ToTensor(),
        Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
    ])

class SimpleTokenizer(object):
    def __init__(self, bpe_path: str):
        self.byte_encoder = bytes_to_unicode()
        self.byte_decoder = {v: k for k, v in self.byte_encoder.items()}
        merges = gzip.open(bpe_path).read().decode("utf-8").split('\n')
        merges = merges[1:49152-256-2+1]
        merges = [tuple(merge.split()) for merge in merges]
        vocab = list(bytes_to_unicode().values())
        vocab = vocab + [v+'</w>' for v in vocab]
        for merge in merges:
            vocab.append(''.join(merge))
        vocab.extend(['<|startoftext|>', '<|endoftext|>'])
        self.encoder = dict(zip(vocab, range(len(vocab))))
        self.decoder = {v: k for k, v in self.encoder.items()}
        self.bpe_ranks = dict(zip(merges, range(len(merges))))
        self.cache = {'<|startoftext|>': '<|startoftext|>', '<|endoftext|>': '<|endoftext|>'}
        self.pat = re.compile(r"""<\|startoftext\|>|<\|endoftext\|>|'s|'t|'re|'ve|'m|'ll|'d|[\p{L}]+|[\p{N}]|[^\s\p{L}\p{N}]+""", re.IGNORECASE)

    def bpe(self, token):
        if token in self.cache:
            return self.cache[token]
        word = tuple(token[:-1]) + ( token[-1] + '</w>',)
        pairs = get_pairs(word)

        if not pairs:
            return token+'</w>'

        while True:
            bigram = min(pairs, key = lambda pair: self.bpe_ranks.get(pair, float('inf')))
            if bigram not in self.bpe_ranks:
                break
            first, second = bigram
            new_word = []
            i = 0
            while i < len(word):
                try:
                    j = word.index(first, i)
                    new_word.extend(word[i:j])
                    i = j
                except:
                    new_word.extend(word[i:])
                    break

                if word[i] == first and i < len(word)-1 and word[i+1] == second:
                    new_word.append(first+second)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            new_word = tuple(new_word)
            word = new_word
            if len(word) == 1:
                break
            else:
                pairs = get_pairs(word)
        word = ' '.join(word)
        self.cache[token] = word
        return word

    def encode(self, text):
        bpe_tokens = []
        text = whitespace_clean(basic_clean(text)).lower()
        for token in re.findall(self.pat, text):
            token = ''.join(self.byte_encoder[b] for b in token.encode('utf-8'))
            bpe_tokens.extend(self.encoder[bpe_token] for bpe_token in self.bpe(token).split(' '))
        return bpe_tokens

    def decode(self, tokens):
        text = ''.join([self.decoder[token] for token in tokens])
        text = bytearray([self.byte_decoder[c] for c in text]).decode('utf-8', errors="replace").replace('</w>', ' ')
        return text

class ViCLIP(nn.Module):
    """docstring for ViCLIP"""

    def __init__(self, tokenizer=None,
                 pretrain=os.path.join(os.path.dirname(os.path.abspath(__file__)), "ViClip-InternVid-10M-FLT.pth"),
                 freeze_text=True):
        super(ViCLIP, self).__init__()
        if tokenizer:
            self.tokenizer = tokenizer
        else:
            self.tokenizer = SimpleTokenizer()
        self.max_txt_l = 32

        self.vision_encoder_name = 'vit_l14'

        self.vision_encoder_pretrained = False
        self.inputs_image_res = 224
        self.vision_encoder_kernel_size = 1
        self.vision_encoder_center = True
        self.video_input_num_frames = 8  # @TODO automate
        self.vision_encoder_drop_path_rate = 0.1
        self.vision_encoder_checkpoint_num = 24
        self.is_pretrain = pretrain
        self.vision_width = 1024
        self.text_width = 768
        self.embed_dim = 768
        self.masking_prob = 0.9

        self.text_encoder_name = 'vit_l14'
        self.text_encoder_pretrained = False  # 'bert-base-uncased'
        self.text_encoder_d_model = 768

        self.text_encoder_vocab_size = 49408

        # create modules.
        self.vision_encoder = self.build_vision_encoder()
        self.text_encoder = self.build_text_encoder()

        self.temp = nn.parameter.Parameter(torch.ones([]) * 1 / 100.0)
        self.temp_min = 1 / 100.0

        if pretrain:
            # logger.info(f"Load pretrained weights from {pretrain}")
            state_dict = torch.load(pretrain, map_location='cpu')['model']
            self.load_state_dict(state_dict)

        # Freeze weights
        if freeze_text:
            self.freeze_text()

    def freeze_text(self):
        """freeze text encoder"""
        for p in self.text_encoder.parameters():
            p.requires_grad = False

    def no_weight_decay(self):
        ret = {"temp"}
        ret.update(
            {"vision_encoder." + k for k in self.vision_encoder.no_weight_decay()}
        )
        ret.update(
            {"text_encoder." + k for k in self.text_encoder.no_weight_decay()}
        )

        return ret

    def forward(self, image, text, raw_text, idx, log_generation=None, return_sims=False):
        """forward and calculate loss.

        Args:
            image (torch.Tensor): The input images. Shape: [B,T,C,H,W].
            text (dict): TODO
            idx (torch.Tensor): TODO

        Returns: TODO

        """
        self.clip_contrastive_temperature()

        vision_embeds = self.encode_vision(image)
        text_embeds = self.encode_text(raw_text)
        if return_sims:
            sims = torch.nn.functional.normalize(vision_embeds, dim=-1) @ \
                   torch.nn.functional.normalize(text_embeds, dim=-1).transpose(0, 1)
            return sims

        # calculate loss

        ## VTC loss
        loss_vtc = self.clip_loss.vtc_loss(
            vision_embeds, text_embeds, idx, self.temp, all_gather=True
        )

        return dict(
            loss_vtc=loss_vtc,
        )

    def encode_vision(self, image, test=False):
        """encode image / videos as features.

        Args:
            image (torch.Tensor): The input images.
            test (bool): Whether testing.

        Returns: tuple.
            - vision_embeds (torch.Tensor): The features of all patches. Shape: [B,T,L,C].
            - pooled_vision_embeds (torch.Tensor): The pooled features. Shape: [B,T,C].

        """
        if image.ndim == 5:
            image = image.permute(0, 2, 1, 3, 4).contiguous()
        else:
            image = image.unsqueeze(2)

        if not test and self.masking_prob > 0.0:
            return self.vision_encoder(
                image, masking_prob=self.masking_prob
            )

        return self.vision_encoder(image)

    def encode_text(self, text):
        """encode text.
        Args:
            text (dict): The output of huggingface's `PreTrainedTokenizer`. contains keys:
                - input_ids (torch.Tensor): Token ids to be fed to a model. Shape: [B,L].
                - attention_mask (torch.Tensor): The mask indicate padded tokens. Shape: [B,L]. 0 is padded token.
                - other keys refer to "https://huggingface.co/docs/transformers/v4.21.2/en/main_classes/tokenizer#transformers.PreTrainedTokenizer.__call__".
        Returns: tuple.
            - text_embeds (torch.Tensor): The features of all tokens. Shape: [B,L,C].
            - pooled_text_embeds (torch.Tensor): The pooled features. Shape: [B,C].

        """
        device = next(self.text_encoder.parameters()).device
        text = self.text_encoder.tokenize(
            text, context_length=self.max_txt_l
        ).to(device)
        text_embeds = self.text_encoder(text)
        return text_embeds

    @torch.no_grad()
    def clip_contrastive_temperature(self, min_val=0.001, max_val=0.5):
        """Seems only used during pre-training"""
        self.temp.clamp_(min=self.temp_min)

    def build_vision_encoder(self):
        """build vision encoder
        Returns: (vision_encoder, vision_layernorm). Each is a `nn.Module`.

        """
        encoder_name = self.vision_encoder_name
        if encoder_name != "vit_l14":
            raise ValueError(f"Not implemented: {encoder_name}")
        vision_encoder = clip_joint_l14(
            pretrained=self.vision_encoder_pretrained,
            input_resolution=self.inputs_image_res,
            kernel_size=self.vision_encoder_kernel_size,
            center=self.vision_encoder_center,
            num_frames=self.video_input_num_frames,
            drop_path=self.vision_encoder_drop_path_rate,
            checkpoint_num=self.vision_encoder_checkpoint_num,
        )
        return vision_encoder

    def build_text_encoder(self):
        """build text_encoder and possibly video-to-text multimodal fusion encoder.
        Returns: nn.Module. The text encoder

        """
        encoder_name = self.text_encoder_name
        if encoder_name != "vit_l14":
            raise ValueError(f"Not implemented: {encoder_name}")
        text_encoder = clip_text_l14(
            pretrained=self.text_encoder_pretrained,
            embed_dim=self.text_encoder_d_model,
            context_length=self.max_txt_l,
            vocab_size=self.text_encoder_vocab_size,
            checkpoint_num=0,
            tokenizer=self.tokenizer,
        )

        return text_encoder

    def get_text_encoder(self):
        """get text encoder, used for text and cross-modal encoding"""
        encoder = self.text_encoder
        return encoder.bert if hasattr(encoder, "bert") else encoder

    def get_text_features(self, input_text, tokenizer, text_feature_dict={}):
        if input_text in text_feature_dict:
            return text_feature_dict[input_text]
        text_template = f"{input_text}"
        with torch.no_grad():
            # text_token = tokenizer.encode(text_template).cuda()
            text_features = self.encode_text(text_template).float()
            text_features /= text_features.norm(dim=-1, keepdim=True)
            text_feature_dict[input_text] = text_features
        return text_features

    def get_vid_features(self, input_frames):
        with torch.no_grad():
            clip_feat = self.encode_vision(input_frames, test=True).float()
            clip_feat /= clip_feat.norm(dim=-1, keepdim=True)
        return clip_feat

    def get_predict_label(self, clip_feature, text_feats_tensor, top=5):
        label_probs = (100.0 * clip_feature @ text_feats_tensor.T).softmax(dim=-1)
        top_probs, top_labels = label_probs.cpu().topk(top, dim=-1)
        return top_probs, top_labels

def load_temp_embed_with_mismatch(temp_embed_old, temp_embed_new, add_zero=True):
    """
    Add/Remove extra temporal_embeddings as needed.
    https://arxiv.org/abs/2104.00650 shows adding zero paddings works.

    temp_embed_old: (1, num_frames_old, 1, d)
    temp_embed_new: (1, num_frames_new, 1, d)
    add_zero: bool, if True, add zero, else, interpolate trained embeddings.
    """
    # TODO zero pad
    num_frms_new = temp_embed_new.shape[1]
    num_frms_old = temp_embed_old.shape[1]
    # logger.info(f"Load temporal_embeddings, lengths: {num_frms_old}-->{num_frms_new}")
    if num_frms_new > num_frms_old:
        if add_zero:
            temp_embed_new[
            :, :num_frms_old
            ] = temp_embed_old  # untrained embeddings are zeros.
        else:
            raise NotImplementedError
            temp_embed_new = interpolate_temporal_pos_embed(temp_embed_old, num_frms_new)
    elif num_frms_new < num_frms_old:
        temp_embed_new = temp_embed_old[:, :num_frms_new]
    else:  # =
        temp_embed_new = temp_embed_old
    return temp_embed_new

MODEL_PATH = 'https://pjlab-gvm-data.oss-cn-shanghai.aliyuncs.com/internvideo/viclip/'
_MODELS = {
    "ViT-L/14": os.path.join(MODEL_PATH, "ViClip-InternVid-10M-FLT.pth"),
}

class QuickGELU(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(1.702 * x)

class ResidualAttentionBlock_vis(nn.Module):
    def __init__(self, d_model, n_head, drop_path=0., attn_mask=None, dropout=0.):
        super().__init__()

        self.drop_path1 = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.drop_path2 = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.attn = nn.MultiheadAttention(d_model, n_head, dropout=dropout)
        self.ln_1 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("drop1", nn.Dropout(dropout)),
            ("c_proj", nn.Linear(d_model * 4, d_model)),
            ("drop2", nn.Dropout(dropout)),
        ]))
        self.ln_2 = nn.LayerNorm(d_model)
        self.attn_mask = attn_mask

    def attention(self, x):
        self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
        return self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]

    def forward(self, x):
        x = x + self.drop_path1(self.attention(self.ln_1(x)))
        x = x + self.drop_path2(self.mlp(self.ln_2(x)))
        return x

class Transformer_vis(nn.Module):
    def __init__(self, width, layers, heads, drop_path=0., checkpoint_num=0, dropout=0.):
        super().__init__()
        dpr = [x.item() for x in torch.linspace(0, drop_path, layers)]
        self.resblocks = nn.ModuleList()
        for idx in range(layers):
            self.resblocks.append(ResidualAttentionBlock_vis(width, heads, drop_path=dpr[idx], dropout=dropout))
        self.checkpoint_num = checkpoint_num

    def forward(self, x):
        for idx, blk in enumerate(self.resblocks):
            if idx < self.checkpoint_num:
                x = checkpoint.checkpoint(blk, x, use_reentrant=False)
            else:
                x = blk(x)
        return x

class VisionTransformer(nn.Module):
    def __init__(
            self, input_resolution, patch_size, width, layers, heads, output_dim=None,
            kernel_size=1, num_frames=8, drop_path=0., checkpoint_num=0, dropout=0.,
            temp_embed=True,
    ):
        super().__init__()
        self.output_dim = output_dim
        self.conv1 = nn.Conv3d(
            3, width,
            (kernel_size, patch_size, patch_size),
            (kernel_size, patch_size, patch_size),
            (0, 0, 0), bias=False
        )
        scale = width ** -0.5
        self.class_embedding = nn.Parameter(scale * torch.randn(width))
        self.positional_embedding = nn.Parameter(scale * torch.randn((input_resolution // patch_size) ** 2 + 1, width))
        self.ln_pre = nn.LayerNorm(width)
        if temp_embed:
            self.temporal_positional_embedding = nn.Parameter(torch.zeros(1, num_frames, width))
        self.transformer = Transformer_vis(
            width, layers, heads, drop_path=drop_path, checkpoint_num=checkpoint_num,
            dropout=dropout)
        self.ln_post = nn.LayerNorm(width)
        if output_dim is not None:
            self.proj = nn.Parameter(torch.empty(width, output_dim))
        else:
            self.proj = None
        self.dropout = nn.Dropout(dropout)

    def get_num_layers(self):
        return len(self.transformer.resblocks)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'positional_embedding', 'class_embedding', 'temporal_positional_embedding'}

    def mask_tokens(self, inputs, masking_prob=0.0):
        B, L, _ = inputs.shape
        # This is different from text as we are masking a fix number of tokens
        Lm = int(masking_prob * L)
        masked_indices = torch.zeros(B, L)
        indices = torch.argsort(torch.rand_like(masked_indices), dim=-1)[:, :Lm]
        batch_indices = (
            torch.arange(masked_indices.shape[0]).unsqueeze(-1).expand_as(indices)
        )
        masked_indices[batch_indices, indices] = 1
        masked_indices = masked_indices.bool()
        return inputs[~masked_indices].reshape(B, -1, inputs.shape[-1])

    def forward(self, x, masking_prob=0.0):
        x = self.conv1(x)  # shape = [*, width, grid, grid]
        B, C, T, H, W = x.shape
        x = x.permute(0, 2, 3, 4, 1).reshape(B * T, H * W, C)
        x = torch.cat(
            [self.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device),
             x], dim=1)  # shape = [*, grid ** 2 + 1, width]
        x = x + self.positional_embedding.to(x.dtype)
        # temporal pos
        cls_tokens = x[:B, :1, :]
        x = x[:, 1:]
        x = rearrange(x, '(b t) n m -> (b n) t m', b=B, t=T)
        if hasattr(self, 'temporal_positional_embedding'):
            if x.size(1) == 1:
                # This is a workaround for unused parameter issue
                x = x + self.temporal_positional_embedding.mean(1)
            else:
                x = x + self.temporal_positional_embedding
        x = rearrange(x, '(b n) t m -> b (n t) m', b=B, t=T)
        if masking_prob > 0.0:
            x = self.mask_tokens(x, masking_prob)
        x = torch.cat((cls_tokens, x), dim=1)
        x = self.ln_pre(x)
        x = x.permute(1, 0, 2)  # BND -> NBD
        x = self.transformer(x)
        x = self.ln_post(x)
        if self.proj is not None:
            x = self.dropout(x[0]) @ self.proj
        else:
            x = x.permute(1, 0, 2)  # NBD -> BND
        return x

def inflate_weight(weight_2d, time_dim, center=True):
    # logger.info(f'Init center: {center}')
    if center:
        weight_3d = torch.zeros(*weight_2d.shape)
        weight_3d = weight_3d.unsqueeze(2).repeat(1, 1, time_dim, 1, 1)
        middle_idx = time_dim // 2
        weight_3d[:, :, middle_idx, :, :] = weight_2d
    else:
        weight_3d = weight_2d.unsqueeze(2).repeat(1, 1, time_dim, 1, 1)
        weight_3d = weight_3d / time_dim
    return weight_3d

def load_state_dict(model, state_dict, input_resolution=224, patch_size=16, center=True):
    state_dict_3d = model.state_dict()
    for k in state_dict.keys():
        if k in state_dict_3d.keys() and state_dict[k].shape != state_dict_3d[k].shape:
            if len(state_dict_3d[k].shape) <= 2:
                # logger.info(f'Ignore: {k}')
                continue
            # logger.info(f'Inflate: {k}, {state_dict[k].shape} => {state_dict_3d[k].shape}')
            time_dim = state_dict_3d[k].shape[2]
            state_dict[k] = inflate_weight(state_dict[k], time_dim, center=center)

    pos_embed_checkpoint = state_dict['positional_embedding']
    embedding_size = pos_embed_checkpoint.shape[-1]
    num_patches = (input_resolution // patch_size) ** 2
    orig_size = int((pos_embed_checkpoint.shape[-2] - 1) ** 0.5)
    new_size = int(num_patches ** 0.5)
    if orig_size != new_size:
        # logger.info(f'Pos_emb from {orig_size} to {new_size}')
        extra_tokens = pos_embed_checkpoint[:1]
        pos_tokens = pos_embed_checkpoint[1:]
        pos_tokens = pos_tokens.reshape(-1, orig_size, orig_size, embedding_size).permute(0, 3, 1, 2)
        pos_tokens = torch.nn.functional.interpolate(
            pos_tokens, size=(new_size, new_size), mode='bicubic', align_corners=False)
        pos_tokens = pos_tokens.permute(0, 2, 3, 1).flatten(0, 2)
        new_pos_embed = torch.cat((extra_tokens, pos_tokens), dim=0)
        state_dict['positional_embedding'] = new_pos_embed

    message = model.load_state_dict(state_dict, strict=False)
    # logger.info(f"Load pretrained weights: {message}")

def clip_joint_l14(
        pretrained=False, input_resolution=224, kernel_size=1,
        center=True, num_frames=8, drop_path=0, checkpoint_num=0,
        dropout=0.,
):
    model = VisionTransformer(
        input_resolution=input_resolution, patch_size=14,
        width=1024, layers=24, heads=16, output_dim=768,
        kernel_size=kernel_size, num_frames=num_frames,
        drop_path=drop_path, checkpoint_num=checkpoint_num,
        dropout=dropout,
    )
    if pretrained:
        if isinstance(pretrained, str):
            model_name = pretrained
        else:
            model_name = "ViT-L/14"
        # logger.info('load pretrained weights')
        state_dict = torch.load(_MODELS[model_name], map_location='cpu')
        load_state_dict(model, state_dict, input_resolution=input_resolution, patch_size=14, center=center)
    return model.eval()


def interpolate_pos_embed_vit(state_dict, new_model):
    key = "vision_encoder.temporal_positional_embedding"
    if key in state_dict:
        vision_temp_embed_new = new_model.state_dict()[key]
        vision_temp_embed_new = vision_temp_embed_new.unsqueeze(2)  # [1, n, d] -> [1, n, 1, d]
        vision_temp_embed_old = state_dict[key]
        vision_temp_embed_old = vision_temp_embed_old.unsqueeze(2)

        state_dict[key] = load_temp_embed_with_mismatch(
            vision_temp_embed_old, vision_temp_embed_new, add_zero=False
        ).squeeze(2)

    key = "text_encoder.positional_embedding"
    if key in state_dict:
        text_temp_embed_new = new_model.state_dict()[key]
        text_temp_embed_new = text_temp_embed_new.unsqueeze(0).unsqueeze(2)  # [n, d] -> [1, n, 1, d]
        text_temp_embed_old = state_dict[key]
        text_temp_embed_old = text_temp_embed_old.unsqueeze(0).unsqueeze(2)

        state_dict[key] = load_temp_embed_with_mismatch(
            text_temp_embed_old, text_temp_embed_new, add_zero=False
        ).squeeze(2).squeeze(0)
    return state_dict

MODEL_PATH = 'https://huggingface.co/laion/CLIP-ViT-L-14-DataComp.XL-s13B-b90K'
_MODELS = {
    "ViT-L/14": os.path.join(MODEL_PATH, "vit_l14_text.pth"),
}

class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)

class ResidualAttentionBlock_tex(nn.Module):
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

class Transformer_tex(nn.Module):
    def __init__(self, width: int, layers: int, heads: int, attn_mask: torch.Tensor = None,
                 checkpoint_num: int = 0):
        super().__init__()
        self.width = width
        self.layers = layers
        self.resblocks = nn.Sequential(*[ResidualAttentionBlock_tex(width, heads, attn_mask) for _ in range(layers)])

        self.checkpoint_num = checkpoint_num

    def forward(self, x: torch.Tensor):
        if self.checkpoint_num > 0:
            segments = min(self.checkpoint_num, len(self.resblocks))
            return checkpoint.checkpoint_sequential(self.resblocks, segments, x)
        else:
            return self.resblocks(x)

class CLIP_TEXT(nn.Module):
    def __init__(
            self,
            embed_dim: int,
            context_length: int,
            vocab_size: int,
            transformer_width: int,
            transformer_heads: int,
            transformer_layers: int,
            checkpoint_num: int,
            tokenizer=None,
    ):
        super().__init__()

        self.context_length = context_length
        self._tokenizer = tokenizer or SimpleTokenizer()

        self.transformer = Transformer_tex(
            width=transformer_width,
            layers=transformer_layers,
            heads=transformer_heads,
            attn_mask=self.build_attention_mask(),
            checkpoint_num=checkpoint_num,
        )

        self.vocab_size = vocab_size
        self.token_embedding = nn.Embedding(vocab_size, transformer_width)
        self.positional_embedding = nn.Parameter(torch.empty(self.context_length, transformer_width))
        self.ln_final = LayerNorm(transformer_width)

        self.text_projection = nn.Parameter(torch.empty(transformer_width, embed_dim))

    def no_weight_decay(self):
        return {'token_embedding', 'positional_embedding'}

    @functools.lru_cache(maxsize=None)
    def build_attention_mask(self):
        # lazily create causal attention mask, with full attention between the vision tokens
        # pytorch uses additive attention mask; fill with -inf
        mask = torch.empty(self.context_length, self.context_length)
        mask.fill_(float("-inf"))
        mask.triu_(1)  # zero out the lower diagonal
        return mask

    def tokenize(self, texts, context_length=77, truncate=True):
        """
        Returns the tokenized representation of given input string(s)
        Parameters
        ----------
        texts : Union[str, List[str]]
            An input string or a list of input strings to tokenize
        context_length : int
            The context length to use; all CLIP models use 77 as the context length
        truncate: bool
            Whether to truncate the text in case its encoding is longer than the context length
        Returns
        -------
        A two-dimensional tensor containing the resulting tokens, shape = [number of input strings, context_length].
        We return LongTensor when torch version is <1.8.0, since older index_select requires indices to be long.
        """
        if isinstance(texts, str):
            texts = [texts]

        sot_token = self._tokenizer.encoder["<|startoftext|>"]
        eot_token = self._tokenizer.encoder["<|endoftext|>"]
        all_tokens = [[sot_token] + self._tokenizer.encode(text) + [eot_token] for text in texts]
        if packaging.version.parse(torch.__version__) < packaging.version.parse("1.8.0"):
            result = torch.zeros(len(all_tokens), context_length, dtype=torch.long)
        else:
            result = torch.zeros(len(all_tokens), context_length, dtype=torch.int)

        for i, tokens in enumerate(all_tokens):
            if len(tokens) > context_length:
                if truncate:
                    tokens = tokens[:context_length]
                    tokens[-1] = eot_token
                else:
                    raise RuntimeError(f"Input {texts[i]} is too long for context length {context_length}")
            result[i, :len(tokens)] = torch.tensor(tokens)

        return result

    def forward(self, text):
        x = self.token_embedding(text)  # [batch_size, n_ctx, d_model]

        x = x + self.positional_embedding
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ self.text_projection

        return x

def clip_text_l14(
        embed_dim=768,
        context_length=77,
        vocab_size=49408,
        transformer_width=768,
        transformer_heads=12,
        transformer_layers=12,
        checkpoint_num=0,
        pretrained=True,
        tokenizer=None,
):
    model = CLIP_TEXT(
        embed_dim,
        context_length,
        vocab_size,
        transformer_width,
        transformer_heads,
        transformer_layers,
        checkpoint_num,
        tokenizer,
    )
    if pretrained:
        if isinstance(pretrained, str) and pretrained != "bert-base-uncased":
            pretrained = _MODELS[pretrained]
        else:
            pretrained = _MODELS["ViT-L/14"]
        # logger.info(f"Load pretrained weights from {pretrained}")
        state_dict = torch.load(pretrained, map_location='cpu')
        if context_length != state_dict["positional_embedding"].size(0):
            # assert context_length < state_dict["positional_embedding"].size(0), "Cannot increase context length."
            # print(f"Resize positional embedding from {state_dict['positional_embedding'].size(0)} to {context_length}")
            if context_length < state_dict["positional_embedding"].size(0):
                state_dict["positional_embedding"] = state_dict["positional_embedding"][:context_length]
            else:
                state_dict["positional_embedding"] = F.pad(
                    state_dict["positional_embedding"],
                    (0, 0, 0, context_length - state_dict["positional_embedding"].size(0)),
                    value=0,
                )

        message = model.load_state_dict(state_dict, strict=False)
        # print(f"Load pretrained weights from {pretrained}: {message}")
    return model.eval()

def build_clip(config):
    model_cls = config.text_encoder.clip_teacher
    model = eval(model_cls)()
    return model