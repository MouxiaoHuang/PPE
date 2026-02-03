from transformers.modeling_outputs import BaseModelOutputWithPast
from transformers.cache_utils import DynamicCache
import torch
import torch.nn as nn
import math
from typing import Optional, List, Union, Tuple
from torch.nn import CrossEntropyLoss
import numpy as np

import transformers
from transformers import Qwen2_5_VLModel
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import Qwen2_5_VLRotaryEmbedding, Qwen2_5_VLCausalLMOutputWithPast, rotate_half

from transformers.cache_utils import Cache
import sys
sys.path.append("..")
from models.chatunivi import CTM
IGNORE_INDEX = -100
import os
import time
import warnings


#============================== token compression related ==============================
def get_image_token_segments(input_ids, image_token_id):
    """get start and end of single-image/multi-image
    """
    input_ids = input_ids.view(-1)
    image_mask = (input_ids == image_token_id)
    image_tokens_indices = image_mask.nonzero(as_tuple=True)[0]

    if image_tokens_indices.numel() == 0:
        return []

    diffs = image_tokens_indices[1:] - image_tokens_indices[:-1]
    split_points = (diffs != 1).nonzero(as_tuple=True)[0]

    segments = []
    prev_idx = 0
    for split_idx in split_points:
        start = image_tokens_indices[prev_idx].item()
        end = image_tokens_indices[split_idx].item()
        segments.append([start, end])
        prev_idx = split_idx + 1

    start = image_tokens_indices[prev_idx].item()
    end = image_tokens_indices[-1].item()
    segments.append([start, end])

    return segments


def merge_tokens(tokens, position_ids, frames, 
                 spatial_compression=False,
                 spatial_clustering_ratio=[1.0],
                 temporal_compression=False,
                 temporal_clustering_ratio=[1.0],
                 use_ppe=False,
                 ppe_k=8
                 ):
    tokens_per_frame = tokens.shape[0] // frames
    tokens = tokens.view(frames, tokens_per_frame, -1)
    position_ids = position_ids.squeeze(1)\
                                .view(3, frames, tokens_per_frame) \
                                .permute(1, 0, 2)
    
    if temporal_compression:
        events = get_time_events(
            image_features=tokens,
            temporal_clustering_ratio=temporal_clustering_ratio
        )

    if spatial_compression:
        num_stages = len(spatial_clustering_ratio)
        features_list = [[] for _ in range(num_stages)]
        pos_list = [[] for _ in range(num_stages)]
        for f in range(frames):
            frame_tokens = tokens[f]  
            frame_pos = position_ids[f]  
            token_dict = {
                'x': frame_tokens.unsqueeze(0), 
                'token_num': frame_tokens.size(0),
                'idx_token': torch.arange(frame_tokens.size(0), device=frame_tokens.device).unsqueeze(0),
                'agg_weight': torch.ones(1, frame_tokens.size(0), 1, device=frame_tokens.device),
                'mask': None,
                'position_ids': frame_pos,
                'ppe_k': ppe_k if use_ppe else None
            }
            for i in range(num_stages):
                if i == 0:
                    token_dict = CTM(sample_ratio=spatial_clustering_ratio[i], k=5)(token_dict)
                else:
                    token_dict = CTM(sample_ratio=spatial_clustering_ratio[i], k=3)(token_dict)
                features_list[i].append(token_dict["x"])
                pos_list[i].append(token_dict["position_ids"].unsqueeze(0))

        merged_tokens = torch.cat([
            torch.cat(layer_tokens, dim=0) for layer_tokens in features_list
        ], dim=1)

        merged_position_ids = torch.cat([
            torch.cat(layer_pos, dim=0) for layer_pos in pos_list
        ], dim=2) 

        if temporal_compression:
            split_size = []
            for i in range(num_stages):
                split_size.append(features_list[i][0].shape[1])
            merged_tokens, merged_position_ids = merge_time_events_vision_token(events=events, cluster_image_features=merged_tokens, 
                                                split_size=split_size, vision_position_ids=merged_position_ids.permute(1, 0, 2),
                                                spatial_clustering_ratio=spatial_clustering_ratio, use_ppe=use_ppe, ppe_k=ppe_k)
        
        merged_tokens = merged_tokens.reshape(-1, merged_tokens.shape[-1])
        merged_position_ids = merged_position_ids.permute(0, 2, 1).reshape(-1, merged_position_ids.shape[1]) 
        merged_position_ids = merged_position_ids.unsqueeze(2).permute(1, 2, 0) 

        return merged_tokens, merged_position_ids


def get_time_events(image_features, temporal_clustering_ratio):
    from collections import OrderedDict
    cls_feature = torch.mean(image_features, dim=1, keepdim=False).unsqueeze(0).clone() 
    token_dict = {
        'x': cls_feature,
        'token_num': cls_feature.size(1),
        'idx_token': torch.arange(cls_feature.size(1))[None, :].repeat(cls_feature.size(0), 1),
        'agg_weight': cls_feature.new_ones(cls_feature.size(0), cls_feature.size(1), 1),
        'mask': None,
        'position_ids': None, 
        'ppe_k': None
    }
    
    if len(temporal_clustering_ratio) == 1:
        down_dict = CTM(sample_ratio=temporal_clustering_ratio[0], k=5)(token_dict)
    else:
        raise ValueError
    
    events = OrderedDict()
    max_len = 0
    for id, i in enumerate(down_dict["idx_token"][0].tolist()):
        if i not in events:
            events[i] = [id]
        else:
            events[i].append(id)
        max_len = len(events[i]) if max_len < len(events[i]) else max_len
    return events


def merge_time_events_vision_token(events, cluster_image_features, split_size, vision_position_ids, 
                                   spatial_clustering_ratio, use_ppe=False, ppe_k=8):
        ctm_features = torch.split(cluster_image_features, split_size, dim=1)
        ctm_position_ids = torch.split(vision_position_ids, split_size, dim=2)
        final_cluster_features, final_cluster_position_ids = [], []

        for key in events:
            cluster_image_features, cluster_position_ids = [], []
            
            for idx, ctm_feature in enumerate(ctm_features):
                ctm_feature = list(torch.unbind(ctm_feature, dim=0))
                cur_image_features = torch.cat([ctm_feature[i].unsqueeze(0) for i in events[key]], dim=1)
                
                ctm_pos_ids = list(torch.unbind(ctm_position_ids[idx], dim=1))
                cur_position_ids = torch.cat([ctm_pos_ids[i].unsqueeze(0) for i in events[key]], dim=2)

                token_dict = {
                    'x': cur_image_features,
                    'token_num': cur_image_features.size(1),
                    'idx_token': torch.arange(cur_image_features.size(1))[None, :].repeat(cur_image_features.size(0), 1),
                    'agg_weight': cur_image_features.new_ones(cur_image_features.size(0), cur_image_features.size(1), 1),
                    'mask': None,
                    'position_ids': cur_position_ids.squeeze(0),
                    'ppe_k': ppe_k if use_ppe else None
                }

                if idx == 0:
                    cur_token_dict = CTM(sample_ratio=spatial_clustering_ratio[idx], k=5)(token_dict)
                else:
                    cur_token_dict = CTM(sample_ratio=spatial_clustering_ratio[idx], k=3)(token_dict)

                cluster_image_features.append(cur_token_dict["x"])
                cluster_position_ids.append(cur_token_dict["position_ids"])

            final_cluster_features.append(torch.cat(cluster_image_features, dim=1)) 
            final_cluster_position_ids.append(torch.cat(cluster_position_ids, dim=1))

        final_cluster_features = torch.cat(final_cluster_features, dim=1)
        final_cluster_position_ids = torch.cat(final_cluster_position_ids, dim=1)

        if use_ppe:
            final_cluster_position_ids = final_cluster_position_ids.view(-1, 3*ppe_k, final_cluster_position_ids.shape[1])
        else:
            final_cluster_position_ids = final_cluster_position_ids.view(-1, 3, final_cluster_position_ids.shape[1])

        return final_cluster_features, final_cluster_position_ids


def merge_tokens_llm(tokens, position_ids, frames,
                     spatial_compression=False,
                     spatial_clustering_ratio=[1.0],
                     temporal_compression=False,
                     temporal_clustering_ratio=[1.0],
                     use_ppe=False,
                     ppe_k=8):
    tokens_per_frame = tokens.shape[0] // frames
    tokens = tokens.view(frames, tokens_per_frame, -1)
    position_ids = position_ids.squeeze(1) \
                                .view(position_ids.shape[0], frames, tokens_per_frame) \
                                .permute(1, 0, 2)
    
    if temporal_compression:
        raise NotImplementedError
    
    if spatial_compression:
        num_stages = len(spatial_clustering_ratio)
        features_list = [[] for _ in range(num_stages)]
        pos_list = [[] for _ in range(num_stages)]

        for f in range(frames):
            frame_tokens = tokens[f]        
            frame_pos = position_ids[f]      

            token_dict = {
                'x': frame_tokens.unsqueeze(0), 
                'token_num': frame_tokens.size(0),
                'idx_token': torch.arange(frame_tokens.size(0), device=frame_tokens.device).unsqueeze(0),
                'agg_weight': torch.ones(1, frame_tokens.size(0), 1, device=frame_tokens.device),
                'mask': None,
                'position_ids': frame_pos,       
                'ppe_k': ppe_k if use_ppe else None
            }

            for i in range(num_stages):
                if i == 0:
                    token_dict = CTM(sample_ratio=spatial_clustering_ratio[i], k=5)(token_dict)
                else:
                    token_dict = CTM(sample_ratio=spatial_clustering_ratio[i], k=3)(token_dict)
                features_list[i].append(token_dict["x"])
                pos_list[i].append(token_dict["position_ids"].unsqueeze(0))

        merged_tokens = torch.cat([
            torch.cat(layer_tokens, dim=0) for layer_tokens in features_list
        ], dim=1)  

        merged_position_ids = torch.cat([
            torch.cat(layer_pos, dim=0) for layer_pos in pos_list
        ], dim=2)

        merged_tokens = merged_tokens.reshape(-1, merged_tokens.shape[-1])
        merged_position_ids = merged_position_ids.permute(0, 2, 1).reshape(-1, merged_position_ids.shape[1])
        merged_position_ids = merged_position_ids.unsqueeze(2).permute(1, 2, 0)

    return merged_tokens, merged_position_ids


def pad_tokens_llm(video_embeds, video_tokens_position_ids, frames):
    total_tokens = video_embeds.shape[0]
    tokens_per_frame = (total_tokens + frames - 1) // frames 
    expected_tokens = tokens_per_frame * frames
    if total_tokens < expected_tokens:
        print(f"[WARN] tokens.shape[0]={total_tokens} not divisible by frames={frames}, padding to {expected_tokens}")
        pad_len = expected_tokens - total_tokens
        # pad video_embeds
        pad_video_embeds = torch.zeros(
            pad_len, video_embeds.shape[1],
            dtype=video_embeds.dtype,
            device=video_embeds.device
        )
        video_embeds = torch.cat([video_embeds, pad_video_embeds], dim=0)
        # pad position_ids
        pad_position_ids = video_tokens_position_ids[:, :, -1:].repeat(1, 1, pad_len)
        video_tokens_position_ids = torch.cat([video_tokens_position_ids, pad_position_ids], dim=2)
    elif total_tokens > expected_tokens:
        print(f"[WARN] tokens.shape[0]={total_tokens} not divisible by frames={frames}, truncating to {expected_tokens}")
        video_embeds = video_embeds[:expected_tokens]
        video_tokens_position_ids = video_tokens_position_ids[:, :, :expected_tokens]
    return video_embeds, video_tokens_position_ids


def merge_tokens_image(tokens, position_ids, attention_mask, labels, image_start_end_list, image_grid_thw,
                       spatial_clustering_ratio=[1.0],
                       use_ppe=False,
                       ppe_k=8,
                       input_ids=None):
    '''only for single image currently, structure is [text, visual, text]'''
    position_ids = position_ids.view(3, -1)     
    attention_mask = attention_mask.view(-1)     
    if labels is not None:
        labels = labels.view(-1)    
    input_ids = input_ids.squeeze(0)     

    num_images = image_grid_thw.shape[0]

    features_list = []
    pos_list = []
    attn_list = []
    labels_list = [] if labels is not None else None
    inp_list = []

    prev_end = -1

    new_image_start_end_list = []

    merged_count = 0

    for img_idx in range(num_images):
        start, end = image_start_end_list[img_idx]

        # 1. keep special tokens before start
        if start > prev_end + 1:
            seg = slice(prev_end + 1, start)
            special_tokens = tokens[seg, :]
            special_pos = position_ids[:, seg]
            special_attn = attention_mask[seg]
            special_labels = labels[seg] if labels is not None else None
            special_inp = input_ids[seg]

            features_list.append(special_tokens)

            if use_ppe and ppe_k > 1:
                special_pos = special_pos.repeat(ppe_k, 1)

            pos_list.append(special_pos)
            attn_list.append(special_attn)
            if labels_list is not None:
                labels_list.append(special_labels)
            inp_list.append(special_inp)

            merged_count += special_tokens.size(0)

        # 2. cluster image tokens
        image_i_tokens = tokens[start:end+1, :]
        image_i_pos = position_ids[:, start:end+1]
        image_i_attn = attention_mask[start:end+1]
        image_i_labels = labels[start:end+1] if labels is not None else None
        image_i_inp = input_ids[start:end+1]

        token_dict = {
            'x': image_i_tokens.unsqueeze(0),
            'token_num': image_i_tokens.size(0),
            'idx_token': torch.arange(image_i_tokens.size(0),
                                    device=image_i_tokens.device).unsqueeze(0),
            'agg_weight': torch.ones(1, image_i_tokens.size(0), 1,
                                    device=image_i_tokens.device),
            'mask': None,
            'position_ids': image_i_pos,
            'ppe_k': ppe_k if use_ppe else None
        }

        img_cluster_start = None

        for i in range(len(spatial_clustering_ratio)):
            if i == 0:
                token_dict = CTM(sample_ratio=spatial_clustering_ratio[i], k=5)(token_dict)
            else:
                token_dict = CTM(sample_ratio=spatial_clustering_ratio[i], k=3)(token_dict)
            new_feats = token_dict["x"].squeeze(0)         
            new_pos   = token_dict["position_ids"].squeeze(0) 
            new_n     = new_feats.size(0)

            if img_cluster_start is None:
                img_cluster_start = merged_count

            features_list.append(new_feats)

            pos_list.append(new_pos)
            attn_list.append(torch.ones(new_n, device=new_feats.device))
            if labels_list is not None:
                labels_list.append(torch.full((new_n,), -100,
                                            device=new_feats.device))
            inp_list.append(torch.full((new_n,), 151655,
                                       device=new_feats.device))

            merged_count += new_n

        img_cluster_end = merged_count - 1
        new_image_start_end_list.append((img_cluster_start, img_cluster_end))

        prev_end = end

    # 3. keep the last part of special tokens
    tail_start = prev_end + 1
    if tail_start < tokens.size(0):
        special_tokens = tokens[tail_start:, :]
        special_pos = position_ids[:, tail_start:]
        special_attn = attention_mask[tail_start:]
        special_labels = labels[tail_start:] if labels is not None else None
        special_inp = input_ids[tail_start:]

        features_list.append(special_tokens)

        if use_ppe and ppe_k > 1:
            special_pos = special_pos.repeat(ppe_k, 1)

        pos_list.append(special_pos)
        attn_list.append(special_attn)
        if labels_list is not None:
            labels_list.append(special_labels)
        inp_list.append(special_inp)

        merged_count += special_tokens.size(0)

    # 4. recover (concat) all parts
    merged_tokens = torch.cat(features_list, dim=0).unsqueeze(0)      
    merged_position_ids = torch.cat(pos_list, dim=1).unsqueeze(1)  
    merged_attention_mask = torch.cat(attn_list, dim=0).unsqueeze(0)  
    merged_labels = (torch.cat(labels_list, dim=0).unsqueeze(0)
                    if labels_list is not None else None)   
    merged_input_ids = torch.cat(inp_list, dim=0).unsqueeze(0)       

    return (merged_tokens,
            merged_position_ids,
            merged_attention_mask,
            merged_labels,
            new_image_start_end_list,
            merged_input_ids,
            token_dict)
#============================== token compression related ==============================


def replace_qwen2_5_with_mixed_modality_forward():
    from transformers import Qwen2_5_VLForConditionalGeneration
    Qwen2_5_VLForConditionalGeneration.forward = qwen2_5_mixed_modality_forward_with_flce
    Qwen2_5_VLModel.forward = qwen2_5_vlmodel_forward
    Qwen2_5_VLRotaryEmbedding.forward = qwen2_5_vlrotaryembedding_forward

    transformers.models.qwen2_5_vl.modeling_qwen2_5_vl.apply_multimodal_rotary_pos_emb = apply_multimodal_rotary_pos_emb_ppe


### rewrite functions to intergrate token compression and PPE
### transformers==4.50.2
def qwen2_5_mixed_modality_forward_with_flce(
    self,
    input_ids: torch.LongTensor = None,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_values: Optional[List[torch.FloatTensor]] = None,
    inputs_embeds: Optional[torch.FloatTensor] = None,
    labels: Optional[torch.LongTensor] = None,
    use_cache: Optional[bool] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    return_dict: Optional[bool] = None,
    pixel_values: Optional[torch.Tensor] = None,
    pixel_values_videos: Optional[torch.FloatTensor] = None,
    image_grid_thw: Optional[torch.LongTensor] = None,
    video_grid_thw: Optional[torch.LongTensor] = None,
    rope_deltas: Optional[torch.LongTensor] = None,
    cache_position: Optional[torch.LongTensor] = None,
    second_per_grid_ts: Optional[torch.Tensor] = None,
    ppe_config: Optional[dict] = None,
) -> Union[Tuple, Qwen2_5_VLCausalLMOutputWithPast]:
    '''rewrite Qwen2_5_VLForConditionalGeneration.forward
    '''
    
    mrope_section = self.config.rope_scaling["mrope_section"]
    ppe_k = ppe_config.get("ppe_k")
    # PPE: update the original mrope section once
    if len(mrope_section) == 3:
        if ppe_config.get("use_ppe") and ppe_k > 1:
            # [16, 24, 24] --> [[t]*K, [h]*K, [w]*K]
            bad = any(x < ppe_k or x % ppe_k != 0 for x in mrope_section)
            if bad:
                # fallback to 1 to contain the most position information in rope embed
                new_mrope_section = [1] * sum(mrope_section)
                msg = (
                    f"\n"
                    f"===============================================================================\n"
                    f"[PPE Warning] ppe_k={ppe_k} cannot evenly split mrope_section={mrope_section}. \n"
                    f"Falling back to minimal split [1,1,...,1] with length={len(new_mrope_section)}.\n"
                    f"==============================================================================="
                )
                warnings.warn(msg, stacklevel=2)
                print(msg)
            else:
                new_mrope_section = []
                for x in mrope_section:
                    new_mrope_section.extend([x // ppe_k] * ppe_k)
            self.config.rope_scaling["mrope_section"] = new_mrope_section
    
    video_start_end = [0, 0] # init

    output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
    output_hidden_states = (
        output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
    )
    return_dict = return_dict if return_dict is not None else self.config.use_return_dict

    if inputs_embeds is None:
        inputs_embeds = self.model.embed_tokens(input_ids)
    
        # Pass dummy image and dummy grid to the visual model to avoid deepspeed error.
        if pixel_values is None and pixel_values_videos is None:
            # Create dummy pixel_values and grid_thw for avoiding deepspeed error.
            dummy_pixel = torch.zeros(14308, 1176).to(self.visual.device)
            dummy_grid = torch.tensor([[1, 98, 146]]).to(self.visual.device)
            
            dummy_pixel = dummy_pixel.type(self.visual.dtype)
            image_embeds = self.visual(dummy_pixel, grid_thw=dummy_grid)
            # Operates as maksed_scatter for the image tokens
            # However the values are all zeros so it dosen't affect the embeddings.
            # This could avoid deepspeed error when some batch only has texts.
            inputs_embeds += image_embeds.mean() * 0
            
        if pixel_values is not None:
            pixel_values = pixel_values.type(self.visual.dtype)
            image_embeds = self.visual(pixel_values, grid_thw=image_grid_thw)
            n_image_tokens = (input_ids == self.config.image_token_id).sum().item()
            n_image_features = image_embeds.shape[0]
            if n_image_tokens != n_image_features:
                raise ValueError(
                    f"Image features and image tokens do not match: tokens: {n_image_tokens}, features {n_image_features}"
                )

            mask = input_ids == self.config.image_token_id
            mask_unsqueezed = mask.unsqueeze(-1)
            mask_expanded = mask_unsqueezed.expand_as(inputs_embeds)
            image_mask = mask_expanded.to(inputs_embeds.device)

            image_embeds = image_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
            inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)

            image_start_end_list = get_image_token_segments(input_ids=input_ids, image_token_id=self.config.image_token_id)

        if pixel_values_videos is not None:
            pixel_values_videos = pixel_values_videos.type(self.visual.dtype)
            video_embeds = self.visual(pixel_values_videos, grid_thw=video_grid_thw)
            n_video_tokens = (input_ids == self.config.video_token_id).sum().item()
            n_video_features = video_embeds.shape[0]
            if n_video_tokens != n_video_features:
                raise ValueError(
                    f"Video features and video tokens do not match: tokens: {n_video_tokens}, features {n_video_features}"
                )

            mask = input_ids == self.config.video_token_id
            ### video start/end
            video_tokens_indices = mask.nonzero(as_tuple=True)[1]
            i_start, i_end = video_tokens_indices[0], video_tokens_indices[-1]
            video_start_end = [i_start, i_end] 
            ###
            mask_unsqueezed = mask.unsqueeze(-1)
            mask_expanded = mask_unsqueezed.expand_as(inputs_embeds)
            video_mask = mask_expanded.to(inputs_embeds.device)

            video_embeds = video_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
            inputs_embeds = inputs_embeds.masked_scatter(video_mask, video_embeds)

        if attention_mask is not None:
            attention_mask = attention_mask.to(inputs_embeds.device)

    # if we get 4D attention mask we cannot calculate rope deltas anymore. TODO @raushan fixme
    if position_ids is None and (attention_mask is None or attention_mask.ndim == 2):
        # calculate RoPE index once per generation in the pre-fill stage only
        if (cache_position is not None and cache_position[0] == 0) or self.rope_deltas is None:
            position_ids, rope_deltas = self.get_rope_index(
                input_ids,
                image_grid_thw,
                video_grid_thw,
                second_per_grid_ts,
                attention_mask,
            )
            self.rope_deltas = rope_deltas
        # then use the prev pre-calculated rope-deltas to get the correct position ids
        else:
            batch_size, seq_length, _ = inputs_embeds.shape
            delta = (
                (cache_position[0] + self.rope_deltas).to(inputs_embeds.device)
                if cache_position is not None
                else 0
            )
            position_ids = torch.arange(seq_length, device=inputs_embeds.device)
            position_ids = position_ids.view(1, -1).expand(batch_size, -1)
            if cache_position is not None:  # otherwise `deltas` is an int `0`
                delta = delta.repeat_interleave(batch_size // delta.shape[0], dim=0)
            position_ids = position_ids.add(delta)
            position_ids = position_ids.unsqueeze(0).expand(3, -1, -1)


    # [VIDEO] [Before LLM] apply visual token compression
    if pixel_values_videos is not None and \
        ppe_config.get("compression_before_llm") and (ppe_config.get("spatial_compression") or ppe_config.get("temporal_compression")):
        assert input_ids.shape[0] == 1, "only supports batch_size == 1 currently"
        if ppe_config.get("temporal_compression") and not ppe_config.get("spatial_compression"):
            print("Currently, temporal_compression cannot work independently")
            raise NotImplementedError
        # Step 1. get video token index
        video_mask = input_ids == self.config.video_token_id
        video_tokens_indices = video_mask.nonzero(as_tuple=True)[1]
        i_start, i_end = video_tokens_indices[0], video_tokens_indices[-1]
        # Step 2. split the corresponding position_ids and do clustering
        video_tokens_position_ids = position_ids.squeeze(1).index_select(1, video_tokens_indices).unsqueeze(1)
        super_frames = video_grid_thw[0][0] # for qwen2.5vl, two frames are treated as one frame
        total_tokens = video_embeds.shape[0]
        tokens_per_frame = (total_tokens + super_frames - 1) // super_frames
        expected_tokens = super_frames * tokens_per_frame
        if total_tokens < expected_tokens:
            print(f"[WARN] tokens.shape[0]={total_tokens} not divisible by super_frames={super_frames}, padding to {expected_tokens}")
            pad_len = expected_tokens - total_tokens
            # pad video_embeds
            pad_video_embeds = torch.zeros(
                pad_len, video_embeds.shape[1],
                dtype=video_embeds.dtype,
                device=video_embeds.device
            )
            video_embeds = torch.cat([video_embeds, pad_video_embeds], dim=0)
            # pad position_ids
            pad_position_ids = video_tokens_position_ids[:, :, -1:].repeat(1, 1, pad_len)
            video_tokens_position_ids = torch.cat([video_tokens_position_ids, pad_position_ids], dim=2)
        elif total_tokens > expected_tokens:
            print(f"[WARN] tokens.shape[0]={total_tokens} not divisible by super_frames={super_frames}, truncating to {expected_tokens}")
            video_embeds = video_embeds[:expected_tokens]
            video_tokens_position_ids = video_tokens_position_ids[:, :, :expected_tokens]
        # Step 3. clustering (merge visual tokens), integrated with PPE
        merged_video_embeds, merged_video_tokens_position_ids = merge_tokens(
            tokens=video_embeds,
            position_ids=video_tokens_position_ids,
            frames=super_frames,
            spatial_compression=ppe_config.get("spatial_compression"),
            spatial_clustering_ratio=ppe_config.get("spatial_clustering_ratio"),
            temporal_compression=ppe_config.get("temporal_compression"),
            temporal_clustering_ratio=ppe_config.get("temporal_clustering_ratio"),
            use_ppe=ppe_config.get("use_ppe"),
            ppe_k=ppe_config.get("ppe_k")
        )
        
        # Step 3. before/after part of video token
        inputs_embeds_visual_before = inputs_embeds[:, :i_start, :]
        inputs_embeds_visual_after = inputs_embeds[:, i_end + 1:, :]
        inputs_embeds = torch.cat(
            [inputs_embeds_visual_before, merged_video_embeds.unsqueeze(0), inputs_embeds_visual_after],
            dim=1
        )
        # calculate new video_start_end
        video_start_end = [i_start, i_start + merged_video_embeds.shape[0] - 1]

        # Step 4. concat new position_ids
        position_ids_visual_before = position_ids[:, :, :i_start]
        position_ids_visual_after = position_ids[:, :, i_end + 1:]
        if ppe_config.get("use_ppe"):
            position_ids_visual_before = position_ids_visual_before.repeat(ppe_config.get("ppe_k"), 1, 1)
            position_ids_visual_after = position_ids_visual_after.repeat(ppe_config.get("ppe_k"), 1, 1)
        position_ids = torch.cat(
            [position_ids_visual_before, merged_video_tokens_position_ids, position_ids_visual_after],
            dim=2
        )

        # Step 5. update attention_mask
        attention_mask_visual_before = attention_mask[:, :i_start]
        attention_mask_visual_after = attention_mask[:, i_end + 1:]
        merged_video_attention_mask = torch.ones(
            (attention_mask.size(0), merged_video_embeds.size(0)),
            dtype=attention_mask.dtype,
            device=attention_mask.device
        )
        attention_mask = torch.cat(
            [attention_mask_visual_before, merged_video_attention_mask, attention_mask_visual_after],
            dim=1
        )

        # Step 6. update labels
        if labels is not None:
            labels_visual_before = labels[:, :i_start]
            labels_visual_after = labels[:, i_end + 1:]
            merged_video_labels = torch.full(
                (labels.size(0), merged_video_embeds.size(0)),
                fill_value=IGNORE_INDEX,
                dtype=labels.dtype,
                device=labels.device
            )
            labels = torch.cat(
                [labels_visual_before, merged_video_labels, labels_visual_after],
                dim=1
            )

        # Step 7. (optional) original chatunivi, calculate position_ids after clustering, 3d rope as default
        if not ppe_config.get("use_ppe"):
            batch_size, seq_length, _ = inputs_embeds.shape
            position_ids = torch.arange(seq_length, device=inputs_embeds.device)
            position_ids = position_ids.view(1, -1).expand(batch_size, -1)
            position_ids = position_ids.unsqueeze(0).expand(3, -1, -1)

    # [IMAGE] [Before LLM] apply visual token compression
    if pixel_values is not None and ppe_config.get("compression_before_llm") and ppe_config.get("spatial_compression"):
        # Step 1. image token index
        assert input_ids.shape[0] == 1, "spatial_cluster mode only supports batch_size == 1"

        # Step 2. there are special tokens between image tokens, thus pass complete inputs_embeds and position_ids in
        inputs_embeds, position_ids, attention_mask, labels, image_start_end_list, new_input_ids, token_dict = merge_tokens_image(
            tokens=inputs_embeds.squeeze(0),
            position_ids=position_ids,
            attention_mask=attention_mask,
            labels=labels,
            image_start_end_list=image_start_end_list,
            image_grid_thw=image_grid_thw,
            spatial_clustering_ratio=ppe_config.get("spatial_clustering_ratio"),
            use_ppe=ppe_config.get("use_ppe"),
            ppe_k=ppe_config.get("ppe_k"),
            input_ids=input_ids
        )

        # Step 3. (optional) original chatunivi, calculate position_ids after clustering, 3d rope as default
        if not ppe_config.get("use_ppe"):
            batch_size, seq_length, _ = inputs_embeds.shape
            position_ids = torch.arange(seq_length, device=inputs_embeds.device)
            position_ids = position_ids.view(1, -1).expand(batch_size, -1)
            position_ids = position_ids.unsqueeze(0).expand(3, -1, -1)

    outputs = self.model(
        input_ids=None,
        position_ids=position_ids, 
        attention_mask=attention_mask,
        past_key_values=past_key_values,
        inputs_embeds=inputs_embeds, 
        use_cache=use_cache,
        output_attentions=output_attentions,
        output_hidden_states=output_hidden_states,
        return_dict=return_dict,
        cache_position=cache_position,
        video_start_end=video_start_end,
        super_frames = video_grid_thw[0][0] if video_grid_thw is not None and video_grid_thw.numel() > 0 else None,
        image_grid_thw = image_grid_thw,
        image_start_end_list = image_start_end_list if pixel_values is not None and ppe_config.get("compression_cascade") else None,
        labels = labels,
        ppe_config=ppe_config
    )
   
    # [IMAGE] cascade
    if pixel_values is not None and ppe_config.get("compression_cascade"):
        outputs, labels = outputs[0], outputs[1]
        raise NotImplementedError
    
    hidden_states = outputs[0]

    if ppe_config.get("compression_cascade") and ppe_config.get("spatial_compression"):
        # [IMAGE] already updated inside LLM (NotImplementedError)

        # [VIDEO] cascade
        if labels is not None and pixel_values_videos is not None:
            labels_visual_before = labels[:, :video_start_end[0]]
            labels_visual_after = labels[:, video_start_end[1] + 1:]
            merged_video_labels = torch.full(
                (labels.size(0), hidden_states.shape[1] - labels_visual_before.shape[1] - labels_visual_after.shape[1]),
                fill_value=IGNORE_INDEX,
                dtype=labels.dtype,
                device=labels.device
            )
            labels = torch.cat(
                [labels_visual_before, merged_video_labels, labels_visual_after],
                dim=1
            )
    
    loss = None
    logits = None

    if os.environ.get("USE_LIGER", "False").lower() == "true":
        # liger kernel branch
        if self.training and (labels is not None):
            from liger_kernel.transformers.fused_linear_cross_entropy import LigerFusedLinearCrossEntropyLoss
            shift_hidden_states = hidden_states[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()

            # Flatten tokens
            shift_hidden_states = shift_hidden_states.view(-1, self.config.hidden_size)
            shift_labels = shift_labels.view(-1)

            lce = LigerFusedLinearCrossEntropyLoss()
            loss = lce(self.lm_head.weight, shift_hidden_states, shift_labels)
        else:
            logits = self.lm_head(hidden_states)
            if labels is not None:
                # Upcast to float if we need to compute the loss to avoid potential precision issues
                logits = logits.float()
                # Shift so that tokens < n predict n
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()
                # Flatten the tokens
                loss_fct = CrossEntropyLoss()
                shift_logits = shift_logits.view(-1, self.config.vocab_size)
                shift_labels = shift_labels.view(-1)
                # Enable model parallelism
                shift_labels = shift_labels.to(shift_logits.device)
                loss = loss_fct(shift_logits, shift_labels)
    else:
        # original branch
        logits = self.lm_head(hidden_states)
        if labels is not None:
            # Upcast to float if we need to compute the loss to avoid potential precision issues
            logits = logits.float()
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

    if not return_dict:
        output = (logits,) + outputs[1:]
        return (loss,) + output if loss is not None else output

    return Qwen2_5_VLCausalLMOutputWithPast(
        loss=loss,
        logits=logits,
        past_key_values=outputs.past_key_values,
        hidden_states=outputs.hidden_states,
        attentions=outputs.attentions,
        rope_deltas=self.rope_deltas,
    )


def qwen2_5_vlmodel_forward(
    self,
    input_ids: torch.LongTensor = None,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_values: Optional[List[torch.FloatTensor]] = None,
    inputs_embeds: Optional[torch.FloatTensor] = None,
    use_cache: Optional[bool] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    return_dict: Optional[bool] = None,
    cache_position: Optional[torch.LongTensor] = None,
    video_start_end: Optional[List] = None,
    super_frames: Optional[int] = None,
    image_grid_thw: Optional[torch.Tensor] = None,
    image_start_end_list: Optional[List] = None,
    labels: Optional[torch.LongTensor] = None,
    ppe_config: Optional[dict] = None,
) -> Union[Tuple, BaseModelOutputWithPast]:
    '''rewrite Qwen2_5_VLModel forward, to insert cascade integration
    '''

    output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
    output_hidden_states = (
        output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
    )
    use_cache = use_cache if use_cache is not None else self.config.use_cache

    return_dict = return_dict if return_dict is not None else self.config.use_return_dict

    if (input_ids is None) ^ (inputs_embeds is not None):
        raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

    if self.gradient_checkpointing and self.training:
        if use_cache:
            # logger.warning_once(
            print(
                "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
            )
            use_cache = False

    # torch.jit.trace() doesn't support cache objects in the output
    if use_cache and past_key_values is None and not torch.jit.is_tracing():
        past_key_values = DynamicCache()

    if inputs_embeds is None:
        inputs_embeds = self.embed_tokens(input_ids)

    if cache_position is None:
        past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
        cache_position = torch.arange(
            past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
        )

    # the hard coded `3` is for temporal, height and width.
    if position_ids is None:
        position_ids = cache_position.view(1, 1, -1).expand(3, inputs_embeds.shape[0], -1)
    elif position_ids.dim() == 2:
        position_ids = position_ids[None, ...].expand(3, position_ids.shape[0], -1)

    causal_mask = self._update_causal_mask(
        attention_mask, inputs_embeds, cache_position, past_key_values, output_attentions
    )

    hidden_states = inputs_embeds

    # create position embeddings to be shared across the decoder layers
    position_embeddings = self.rotary_emb(hidden_states, position_ids)

    # decoder layers
    all_hidden_states = () if output_hidden_states else None
    all_self_attns = () if output_attentions else None
    next_decoder_cache = None

    # for decoder_layer in self.layers:
    for layer_idx, decoder_layer in enumerate(self.layers):
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        # [IMAGE] cascade
        if ppe_config.get("compression_cascade", False) and \
            image_grid_thw is not None and layer_idx in ppe_config.get("cascade_layer", []):
            raise NotImplementedError

        # [VIDEO] cascade
        if ppe_config.get("compression_cascade", False) and \
            video_start_end is not None and layer_idx in ppe_config.get("cascade_layer", []) and video_start_end[0] != video_start_end[1]:
            s, e = video_start_end[0], video_start_end[1]
            before = hidden_states[:, :s, :]
            video_part = hidden_states[:, s:e+1, :]
            after = hidden_states[:, e+1:, :]

            pos_before = position_ids[:, :, :s]
            pos_video = position_ids[:, :, s:e+1]
            pos_after = position_ids[:, :, e+1:]
            
            video_part, pos_video = pad_tokens_llm(video_embeds=video_part.squeeze(0),
                                                video_tokens_position_ids=pos_video,
                                                frames=super_frames)
            
            merged_v, merged_p = merge_tokens_llm(video_part, pos_video, super_frames, 
                                                spatial_compression=ppe_config.get("spatial_compression"),
                                                spatial_clustering_ratio=ppe_config.get("spatial_clustering_ratio"),
                                                temporal_compression=ppe_config.get("temporal_compression"),
                                                temporal_clustering_ratio=ppe_config.get("temporal_clustering_ratio"),
                                                use_ppe=ppe_config.get("use_ppe"),
                                                ppe_k=ppe_config.get("ppe_k"))

            hidden_states = torch.cat([before, merged_v.unsqueeze(0), after], dim=1)

            if position_ids.shape[0] == 3:
                if ppe_config.get("ppe_k") > 1:
                    pos_before = pos_before.repeat(ppe_config.get("ppe_k"), 1, 1)
                    pos_after = pos_after.repeat(ppe_config.get("ppe_k"), 1, 1)

            position_ids = torch.cat([pos_before, merged_p, pos_after], dim=2)

            attention_mask_visual_before = attention_mask[:, :s]
            attention_mask_visual_after = attention_mask[:, e+1:]
            merged_video_attention_mask = torch.ones(
                (attention_mask.size(0), merged_v.size(0)),
                dtype=attention_mask.dtype,
                device=attention_mask.device
            )
            attention_mask = torch.cat(
                [attention_mask_visual_before, merged_video_attention_mask, attention_mask_visual_after],
                dim=1
            )

            video_start_end = [s, s + merged_v.shape[0] - 1]

            position_embeddings = self.rotary_emb(hidden_states, position_ids)
            causal_mask = self._update_causal_mask(
                attention_mask, hidden_states, cache_position, past_key_values, output_attentions
            )

        if self.gradient_checkpointing and self.training:
            layer_outputs = self._gradient_checkpointing_func(
                decoder_layer.__call__,
                hidden_states,
                causal_mask, 
                position_ids, 
                past_key_values, 
                output_attentions,
                use_cache,
                cache_position, 
                position_embeddings, 
            )
        else:
            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=causal_mask,
                position_ids=position_ids,
                past_key_value=past_key_values,
                output_attentions=output_attentions,
                use_cache=use_cache,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
            )

        hidden_states = layer_outputs[0]

        if use_cache:
            next_decoder_cache = layer_outputs[2 if output_attentions else 1]

        if output_attentions:
            all_self_attns += (layer_outputs[1],)

    hidden_states = self.norm(hidden_states)

    # add hidden states from the last decoder layer
    if output_hidden_states:
        all_hidden_states += (hidden_states,)

    next_cache = next_decoder_cache if use_cache else None

    if image_grid_thw is not None and image_start_end_list is not None: # only when compressing image tokens
        if not return_dict:
            outputs = tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns] if v is not None)
            return outputs, labels
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        ), labels

    else:
        if not return_dict:
            return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns] if v is not None)
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )


def qwen2_5_vlrotaryembedding_forward(self, x, position_ids):
    '''rewrite, change the hard code 3 to position_ids.shape[0]'''

    if "dynamic" in self.rope_type:
        self._dynamic_frequency_update(position_ids, device=x.device)

    # Core RoPE block. In contrast to other models, Qwen2_5_VL has different position ids for the grids
    # So we expand the inv_freq to shape (3, ...)
    # inv_freq_expanded = self.inv_freq[None, None, :, None].float().expand(3, position_ids.shape[1], -1, 1)
    ###@@@ [UPDATE] do not use the hard code 3
    inv_freq_expanded = self.inv_freq[None, None, :, None].float().expand(position_ids.shape[0], position_ids.shape[1], -1, 1)
    position_ids_expanded = position_ids[:, :, None, :].float()  # shape (3, bs, 1, positions)
    # Force float32 (see https://github.com/huggingface/transformers/pull/29285)
    device_type = x.device.type
    device_type = device_type if isinstance(device_type, str) and device_type != "mps" else "cpu"
    with torch.autocast(device_type=device_type, enabled=False):
        freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(2, 3)
        emb = torch.cat((freqs, freqs), dim=-1)
        cos = emb.cos()
        sin = emb.sin()

    # Advanced RoPE types (e.g. yarn) apply a post-processing scaling factor, equivalent to scaling attention
    cos = cos * self.attention_scaling
    sin = sin * self.attention_scaling

    return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


def apply_multimodal_rotary_pos_emb_ppe(q, k, cos, sin, mrope_section, unsqueeze_dim=1):
    '''rewrite mrope, replace the original version
    '''
    mrope_section = mrope_section * 2
    cos = torch.cat([m[i % cos.shape[0]] for i, m in enumerate(cos.split(mrope_section, dim=-1))], dim=-1).unsqueeze(
        unsqueeze_dim
    )
    sin = torch.cat([m[i % sin.shape[0]] for i, m in enumerate(sin.split(mrope_section, dim=-1))], dim=-1).unsqueeze(
        unsqueeze_dim
    )

    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed