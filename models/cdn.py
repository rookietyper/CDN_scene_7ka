import copy
from typing import Optional, List

import torch
import torch.nn.functional as F
from torch import nn, Tensor
import numpy


class CDN(nn.Module):

    def __init__(self, d_model=512, nhead=8, num_encoder_layers=6,
                 num_dec_layers_hopd=3, num_dec_layers_interaction=3, 
                 dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False,
                 return_intermediate_dec=False, args = None): 
        super().__init__()

        encoder_layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward,
                                                dropout, activation, normalize_before)
        encoder_norm = nn.LayerNorm(d_model) if normalize_before else None
        self.encoder = TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)

        decoder_layer = TransformerDecoderLayer(d_model, nhead, dim_feedforward,
                                                dropout, activation, normalize_before)
        decoder_norm = nn.LayerNorm(d_model)
        self.decoder = TransformerDecoder(decoder_layer, num_dec_layers_hopd, decoder_norm,
                                          return_intermediate=return_intermediate_dec)


        interaction_decoder_layer = TransformerDecoderLayer(d_model, nhead, dim_feedforward,
                                                dropout, activation, normalize_before)
        interaction_decoder_norm = nn.LayerNorm(d_model)
        self.interaction_decoder = TransformerDecoder(interaction_decoder_layer, num_dec_layers_interaction, interaction_decoder_norm,
                                            return_intermediate=return_intermediate_dec)

        self._reset_parameters()

        self.d_model = d_model
        self.nhead = nhead
        self.use_hier_beforeHOPD = args.use_hier_beforeHOPD 
        self.use_place365_pred_hier2 = args.use_place365_pred_hier2
        if self.use_place365_pred_hier2:
            self.text_proj = nn.Linear(16, 256)
        self.use_place365_pred_hier3 = args.use_place365_pred_hier3
        if self.use_place365_pred_hier3:
            self.text_proj = nn.Linear(512, 512)
        self.use_place365_pred_hier2reclass = args.use_place365_pred_hier2reclass
        if self.use_place365_pred_hier2reclass:
            self.text_proj = nn.Linear(33, 256)
        self.use_background = (self.use_place365_pred_hier2 or self.use_place365_pred_hier3 or self.use_place365_pred_hier2reclass)
        self.use_panoptic_info = args.use_coco_panoptic_info
        self.use_panoptic_num_info = args.use_coco_panoptic_num_info
        self.use_panoptic_info_beforeHOPD = args.use_panoptic_info_beforeHOPD
        if self.use_panoptic_info:
            self.panoptic_proj = nn.Linear(133, 256)
        self.use_panoptic_info_attention = args.use_panoptic_info_attention
        if self.use_panoptic_info_attention:
            self.HOPD_panoptic_embedding = torch.load('./panoptic_stuff_VIT-B32.pth')
            self.panoptic_proj = nn.Linear(512, 256)
            HOPDquery_decoder_layer = HOPDquery_DecoderLayer(d_model, nhead, dim_feedforward,
                                                dropout, activation, normalize_before)
            HOPDquery_decoder_norm = nn.LayerNorm(d_model)
            self.HOPD_panoptic_attention = HOPDquery_Decoder(HOPDquery_decoder_layer, 1, HOPDquery_decoder_norm,
                                            return_intermediate=return_intermediate_dec)
        self.use_CMA = args.use_CMA
        if self.use_CMA:
            self.HOPD_panoptic_embedding = torch.load('./panoptic_stuff_VIT-B32.pth')
            self.hier2_clip_embedding =  torch.load('./hier2_clip_textembedding.pt')
            self.panoptic_proj = nn.Linear(512, 256)
            CMA_decoder_layer = HOPDquery_DecoderLayer(d_model, nhead, dim_feedforward,
                                                dropout, activation, normalize_before)
            CMA_decoder_norm = nn.LayerNorm(d_model)
            self.CMA_attention = HOPDquery_Decoder(CMA_decoder_layer, 1, CMA_decoder_norm,
                                            return_intermediate=return_intermediate_dec)
        
    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def process_encoded_feature(self, memory, mask, background, pos_embed, bs): #用于引入背景信息
        mask_list = mask.tolist()
        if self.use_place365_pred_hier2:
            background = self.text_proj(background.cuda()).unsqueeze(0)
            pos_zero_embed = torch.zeros((1, bs, self.d_model)).cuda()
            for i in range(bs):
              mask_list[i].append(False)
        elif self.use_place365_pred_hier2reclass:
            background = self.text_proj(background.cuda()).unsqueeze(0)
            pos_zero_embed = torch.zeros((1, bs, self.d_model)).cuda()
            for i in range(bs):
              mask_list[i].append(False)
        elif self.use_place365_pred_hier3:
            background = self.text_proj(background.cuda()).unsqueeze(1).reshape(2,bs,-1)
            pos_zero_embed = torch.zeros((2, bs, self.d_model)).cuda()
            for i in range(bs):
              mask_list[i].append(False)
              mask_list[i].append(False)
        
        mask = torch.tensor(mask_list).cuda()
        #print(memory.shape)
        memory = torch.cat((memory, background), dim=0)
        #print(memory.shape)
        
        pos_embed = torch.cat((pos_embed, pos_zero_embed), dim=0)
        
        return memory, mask, pos_embed

    def process_encoded_panoptic_feature(self, memory, mask, panoptic, pos_embed, bs): #用于引入背景信息
        mask_list = mask.tolist()
        panoptic = self.panoptic_proj(panoptic.cuda()).unsqueeze(0)
        pos_zero_embed = torch.zeros((1, bs, self.d_model)).cuda()
        for i in range(bs):
            mask_list[i].append(False)
        
        mask = torch.tensor(mask_list).cuda()
        #print(memory.shape)
        memory = torch.cat((memory, panoptic), dim=0)
        #print(memory.shape)
        
        pos_embed = torch.cat((pos_embed, pos_zero_embed), dim=0)
        
        return memory, mask, pos_embed


    def forward(self, src, mask, query_embed, pos_embed, background = None):
        bs, c, h, w = src.shape
        src = src.flatten(2).permute(2, 0, 1)
        pos_embed = pos_embed.flatten(2).permute(2, 0, 1)
        query_embed = query_embed.unsqueeze(1).repeat(1, bs, 1)
        mask = mask.flatten(1)

        tgt = torch.zeros_like(query_embed)

        memory = self.encoder(src, src_key_padding_mask=mask, pos=pos_embed)# (h*w)*bs*256
        if self.use_CMA: #利用panoptic得到的信息和场景类别信息来增强编码后的特征
            panoptic_info = background[1] # bs * 133
            background = background[0] # bs *16
            panoptic_features = self.panoptic_proj(torch.tensor(self.HOPD_panoptic_embedding,dtype = torch.float32).cuda())# 133 *256
            panoptic_features = panoptic_features.unsqueeze(1).repeat(1,bs,1)  # 133*bs*256
            hier2label_features = self.panoptic_proj(torch.tensor(self.hier2_clip_embedding,dtype = torch.float32).cuda()) # 16*256
            hier2label_features = hier2label_features.unsqueeze(1).repeat(1,bs,1)# 16*bs*256
            final_features = torch.cat((panoptic_features, hier2label_features),dim=0)
            mask_panoptic = ~(panoptic_info==1)
            mask_hier2 = ~(background>0.4)
            mask_CMA=torch.cat((mask_panoptic,mask_hier2),dim=-1) # bs * 16+133
            memory = self.CMA_attention(memory, final_features, tgt_mask = mask, memory_key_padding_mask=mask_CMA)  # 注意这块变了0值放在pos的位置上了
            # torch.Size([1, 725, 2, 256])
            memory = memory[-1]
            # interaction_query_embed = interaction_query_embed.transpose(1, 2)[-1].permute(1, 0, 2)


        if (self.use_panoptic_info or self.use_panoptic_info_attention) and not self.use_CMA:
            if self.use_panoptic_num_info:
                panoptic_info = background[2]
            else:
                panoptic_info = background[1]
            background = background[0]
        if self.use_panoptic_info and self.use_panoptic_info_beforeHOPD:
            memory,mask,pos_embed =  self.process_encoded_panoptic_feature(memory,mask, panoptic_info, pos_embed, bs)
        if self.use_background and self.use_hier_beforeHOPD:
            #print(len((memory,mask,background,pos_embed)))
            memory,mask,pos_embed =  self.process_encoded_feature(memory,mask, background, pos_embed, bs)
        
        
        hopd_out = self.decoder(tgt, memory, memory_key_padding_mask=mask,
                          pos=pos_embed, query_pos=query_embed)
        # print((tgt.shape,memory.shape,mask.shape,pos_embed.shape,query_embed.shape))
        #(torch.Size([64, 1, 256]), torch.Size([589, 1, 256]), torch.Size([1, 589]), torch.Size([589, 1, 256]), torch.Size([64, 1, 256]))
  
        hopd_out = hopd_out.transpose(1, 2)
        
        if self.use_background and not self.use_hier_beforeHOPD:
            memory,mask,pos_embed =  self.process_encoded_feature(memory,mask,background,pos_embed,bs)
        if self.use_panoptic_info and not self.use_panoptic_info_beforeHOPD:
            # print("panoptic after HOPD")
            memory,mask,pos_embed =  self.process_encoded_panoptic_feature(memory,mask, panoptic_info, pos_embed, bs)
        
        
        
        if self.use_panoptic_info_attention:
            interaction_query_embed = hopd_out[-1].permute(1, 0, 2)
            interaction_zeros = torch.zeros_like(interaction_query_embed)
            panoptic_features = self.panoptic_proj(torch.tensor(self.HOPD_panoptic_embedding,dtype = torch.float32).cuda())
            panoptic_features = panoptic_features.unsqueeze(1).repeat(1,bs,1)
            # panoptic_features = panoptic_features*background # 这里后面可能还是得用掩码
            
            mask_panoptic = ~(panoptic_info==1) # bs,133
            interaction_query_embed = self.HOPD_panoptic_attention(interaction_query_embed, panoptic_features, memory_key_padding_mask=mask_panoptic,
                                   query_pos=interaction_zeros)  # 注意这块变了0值放在pos的位置上了
            interaction_query_embed = interaction_query_embed.transpose(1, 2)[-1].permute(1, 0, 2)
            # interaction_query_embed = self.HOPD_panoptic_attention(interaction_zeros, panoptic_features, memory_key_padding_mask=mask_panoptic,
            #                        query_pos=interaction_query_embed)[-1].permute(1, 0, 2)
        else:
            interaction_query_embed = hopd_out[-1].permute(1, 0, 2)

        
        interaction_tgt = torch.zeros_like(interaction_query_embed)
        interaction_decoder_out = self.interaction_decoder(interaction_tgt, memory, memory_key_padding_mask=mask,
                                  pos=pos_embed, query_pos=interaction_query_embed)
        interaction_decoder_out = interaction_decoder_out.transpose(1, 2)

        return hopd_out, interaction_decoder_out


class TransformerEncoder(nn.Module):

    def __init__(self, encoder_layer, num_layers, norm=None):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, src,
                mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None):
        output = src

        for layer in self.layers:
            output = layer(output, src_mask=mask,
                           src_key_padding_mask=src_key_padding_mask, pos=pos)

        if self.norm is not None:
            output = self.norm(output)

        return output


class HOPDquery_Decoder(nn.Module):

    def __init__(self, decoder_layer, num_layers, norm=None, return_intermediate=False):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm
        self.return_intermediate = return_intermediate

    def forward(self, tgt, memory,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
        output = tgt

        intermediate = []

        for layer in self.layers:
            output = layer(output, memory, tgt_mask=tgt_mask,
                           memory_mask=memory_mask,
                           tgt_key_padding_mask=tgt_key_padding_mask,
                           memory_key_padding_mask=memory_key_padding_mask,
                           pos=pos, query_pos=query_pos)
            if self.return_intermediate:
                intermediate.append(self.norm(output))

        if self.norm is not None:
            output = self.norm(output)
            if self.return_intermediate:
                intermediate.pop()
                intermediate.append(output)

        if self.return_intermediate:
            return torch.stack(intermediate)

        return output

class HOPDquery_DecoderLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False):
        super().__init__()
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self, tgt, panoptic_info,
                     tgt_mask: Optional[Tensor] = None,
                     memory_mask: Optional[Tensor] = None,
                     tgt_key_padding_mask: Optional[Tensor] = None,
                     memory_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None,
                     query_pos: Optional[Tensor] = None):
        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt, query_pos),
                                   key=self.with_pos_embed(panoptic_info, pos),
                                   value=panoptic_info, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm2(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm3(tgt)
        return tgt

    def forward(self, tgt, memory,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
        return self.forward_post(tgt, memory, tgt_mask, memory_mask,
                                 tgt_key_padding_mask, memory_key_padding_mask, pos, query_pos)


class TransformerDecoder(nn.Module):

    def __init__(self, decoder_layer, num_layers, norm=None, return_intermediate=False):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm
        self.return_intermediate = return_intermediate

    def forward(self, tgt, memory,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
        output = tgt

        intermediate = []

        for layer in self.layers:
            output = layer(output, memory, tgt_mask=tgt_mask,
                           memory_mask=memory_mask,
                           tgt_key_padding_mask=tgt_key_padding_mask,
                           memory_key_padding_mask=memory_key_padding_mask,
                           pos=pos, query_pos=query_pos)
            if self.return_intermediate:
                intermediate.append(self.norm(output))

        if self.norm is not None:
            output = self.norm(output)
            if self.return_intermediate:
                intermediate.pop()
                intermediate.append(output)

        if self.return_intermediate:
            return torch.stack(intermediate)

        return output


class TransformerEncoderLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self,
                     src,
                     src_mask: Optional[Tensor] = None,
                     src_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None):
        q = k = self.with_pos_embed(src, pos)
        src2 = self.self_attn(q, k, value=src, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src

    def forward_pre(self, src,
                    src_mask: Optional[Tensor] = None,
                    src_key_padding_mask: Optional[Tensor] = None,
                    pos: Optional[Tensor] = None):
        src2 = self.norm1(src)
        q = k = self.with_pos_embed(src2, pos)
        src2 = self.self_attn(q, k, value=src2, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src2 = self.norm2(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src2))))
        src = src + self.dropout2(src2)
        return src

    def forward(self, src,
                src_mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None):
        if self.normalize_before:
            return self.forward_pre(src, src_mask, src_key_padding_mask, pos)
        return self.forward_post(src, src_mask, src_key_padding_mask, pos)


class TransformerDecoderLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self, tgt, memory,
                     tgt_mask: Optional[Tensor] = None,
                     memory_mask: Optional[Tensor] = None,
                     tgt_key_padding_mask: Optional[Tensor] = None,
                     memory_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None,
                     query_pos: Optional[Tensor] = None):
        q = k = self.with_pos_embed(tgt, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)
        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt, query_pos),
                                   key=self.with_pos_embed(memory, pos),
                                   value=memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        return tgt

    def forward_pre(self, tgt, memory,
                    tgt_mask: Optional[Tensor] = None,
                    memory_mask: Optional[Tensor] = None,
                    tgt_key_padding_mask: Optional[Tensor] = None,
                    memory_key_padding_mask: Optional[Tensor] = None,
                    pos: Optional[Tensor] = None,
                    query_pos: Optional[Tensor] = None):
        tgt2 = self.norm1(tgt)
        q = k = self.with_pos_embed(tgt2, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt2, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt2 = self.norm2(tgt)
        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt2, query_pos),
                                   key=self.with_pos_embed(memory, pos),
                                   value=memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt2 = self.norm3(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout3(tgt2)
        return tgt

    def forward(self, tgt, memory,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
        if self.normalize_before:
            return self.forward_pre(tgt, memory, tgt_mask, memory_mask,
                                    tgt_key_padding_mask, memory_key_padding_mask, pos, query_pos)
        return self.forward_post(tgt, memory, tgt_mask, memory_mask,
                                 tgt_key_padding_mask, memory_key_padding_mask, pos, query_pos)


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def build_cdn(args):
    return CDN(
        d_model=args.hidden_dim,
        dropout=args.dropout,
        nhead=args.nheads,
        dim_feedforward=args.dim_feedforward,
        num_encoder_layers=args.enc_layers,
        num_dec_layers_hopd=args.dec_layers_hopd,
        num_dec_layers_interaction=args.dec_layers_interaction,
        normalize_before=args.pre_norm,
        return_intermediate_dec=True,
        args = args

    )


def _get_activation_fn(activation):
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")
