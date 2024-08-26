import json
import copy
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from config import Config
from compute_z import compute_z
from torch.optim.lr_scheduler import CosineAnnealingLR
from util import nethook

import torch.optim as optim
from tqdm import tqdm
import argparse
import random
import numpy as np
import os
from transformers.modeling_attn_mask_utils import AttentionMaskConverter,_prepare_4d_causal_attention_mask

def set_seed(seed=2024):
    '''Sets the seed of the entire notebook so results are the same every time we run.
    This is for REPRODUCIBILITY.'''
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # When running on the CuDNN backend, two further options must be set
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # Set a fixed value for the hash seed
    os.environ['PYTHONHASHSEED'] = str(seed)
    

def compute_ks(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    batch_data: list,
    config: Config,
    layer: int,
):
    input_ids = tok(batch_data, padding=True,return_tensors="pt").to(f"cuda:{config.device}")
    idxs = [i.sum()-1 for i in input_ids['attention_mask']]

    with torch.no_grad():
        with nethook.Trace(
            module=model,
            layer=config.layer_module_tmp.format(layer),
            retain_input=True,
            retain_output=True,
            detach=True,
            clone=True,
            ) as tr:
                _ = model(**input_ids)
                #layer_in_ks = tr.input #(bs:seq:h_dim)
                zs_out = tr.output#(bs:seq:h_dim)
    zs_out = zs_out[0] if type(zs_out) is tuple else zs_out
    zs_out_list=[]
    for i in range(len(zs_out)):
        zs_out_list.append(zs_out[i,idxs[i]])
    zs_out =torch.stack(zs_out_list,dim=0)


    return zs_out,idxs

def get_optimizer_params(model, encoder_lr, weight_decay=0.01):
        param_optimizer = list(model.named_parameters())
        no_decay = ["input_layernorm.weight", "post_attention_layernorm.weight"]
        optimizer_parameters = [
            {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            'lr': encoder_lr, 'weight_decay': weight_decay},
            {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            'lr': encoder_lr, 'weight_decay': 0.0},
        ]
        return optimizer_parameters

def execute_batch_unke(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    config:Config,
    batch_data:list,
    ex_data:list):

    preserve_params = []
    for name, params in model.named_parameters():
        #print(name)
        splitted_name = name.split('.')
        if len(splitted_name) >= 4 and str.isdigit(splitted_name[2]):
            if int(splitted_name[2]) in config.layers:
                preserve_params.append(name)

    #print(preserve_params)
    weights = {
        param: nethook.get_parameter(
            model, param)
        for param in preserve_params
    }
    
    weights_copy = {k: v.detach().clone() for k, v in weights.items()}




    z_layer = config.layers[-1]
    z_list = []
    for data in batch_data:
        
        cur_z = compute_z(   
            model,
            tok,
            data,
            z_layer,
            config,
        )

        z_list.append(cur_z)
    zs = torch.stack(z_list, dim=0)#(bs,h_dim)
    #print(zs.shape)
    batch_question = [i['question'] for i in batch_data]
    # Insert
    for i, layer in enumerate(config.layers):
        #print(f"\n\nLAYER {layer}\n")
        contexts_tok = tok(batch_question, padding=True, return_tensors="pt").to(
            next(model.parameters()).device
        )
        with torch.no_grad():
            with nethook.Trace(
                module=model,
                layer=config.layer_module_tmp.format(layer),
                retain_input=True,
                retain_output=True,
                detach=True,
                clone=True,
            ) as tr:
                _ = model(**contexts_tok)
                layer_in_ks = tr.input #(bs:seq:h_dim)
                layer_out_ks = tr.output#(bs:seq:h_dim)
        layer_out_ks = layer_out_ks[0] if type(layer_out_ks) is tuple else layer_out_ks
        

        cur_zs,idxs = compute_ks(model, tok,batch_question, config, z_layer)
        
        
        targets = zs - cur_zs 


        ex_tok = tok(ex_data, padding=True, return_tensors="pt").to(
            next(model.parameters()).device
        )
        
        with torch.no_grad():
            with nethook.Trace(
                module=model,
                layer=config.layer_module_tmp.format(layer),
                retain_input=True,
                retain_output=True,
                detach=True,
                clone=True,
            ) as tr:
                _ = model(**ex_tok)
                stat_in = tr.input
                stat_out = tr.output
        stat_out = stat_out[0] if type(stat_out) is tuple else stat_out



        resid = targets / (len(config.layers) - i)  # Distribute residual across layers(1,4096)

        
        criterion = nn.MSELoss()
        
        _layer = nethook.get_module(model, config.layer_module_tmp.format(layer))
        
        for n,m in _layer.named_parameters():
            
            m.requires_grad=True
            
        params = get_optimizer_params(_layer,config.lr)
        
        
        optimizer = optim.AdamW(params,lr=config.lr,eps=1e-8,betas = (0.9,0.999))
        
        for i in range(len(idxs)):
            
            layer_out_ks[i,idxs[i]]+=resid[i]
        
        
        # llama2
        if config.model_name == 'LLama2-7B-Chat':
            input_causal_mask,input_position_ids,input_cache_position = get_causal_mask(layer_in_ks,contexts_tok['attention_mask'])
            ex_causal_mask,ex_position_ids,ex_cache_position = get_causal_mask(stat_in,ex_tok['attention_mask'])
        elif config.model_name == 'Qwen1.5-7B-Chat':
            input_causal_mask,input_position_ids = get_qwen2_causal_mask(layer_in_ks,contexts_tok['attention_mask'])
            ex_causal_mask,ex_position_ids = get_qwen2_causal_mask(stat_in,ex_tok['attention_mask'])

       
        for step in range(config.optim_num_step):
            optimizer.zero_grad()
            if config.model_name == 'Qwen1.5-7B-Chat':
                loss = criterion(_layer(stat_in,attention_mask=ex_causal_mask,position_ids=ex_position_ids)[0], stat_out)+ criterion(_layer(layer_in_ks,attention_mask=input_causal_mask,position_ids=input_position_ids)[0], layer_out_ks)
            elif config.model_name == 'LLama2-7B-Chat':
                loss = criterion(_layer(stat_in,attention_mask=ex_causal_mask,position_ids=ex_position_ids,cache_position = ex_cache_position)[0], stat_out)+ criterion(_layer(layer_in_ks,attention_mask=input_causal_mask,position_ids=input_position_ids,cache_position=input_cache_position)[0], layer_out_ks)
            loss.backward(retain_graph=True)
            optimizer.step()    
            
            #print('Step [{}/{}], Loss: {:.4f}, Layer:{}'.format(step+1, config.optim_num_step, loss.item(),layer))
            # if loss.item() < 5e-5:
            #     break

        for x in [layer_in_ks, layer_out_ks,cur_zs, targets,stat_in,stat_out]:
            x.cpu()
            del x
        torch.cuda.empty_cache()
        
    return weights_copy
def get_qwen2_causal_mask(input_tensor,attention_mask,past_key_values_length = 0):
    device = input_tensor.device
    seq_length = input_tensor.shape[1]
    position_ids = torch.arange(
        past_key_values_length, seq_length + past_key_values_length, dtype=torch.long, device=device
    )
    position_ids = position_ids.unsqueeze(0).view(-1, seq_length)

    attention_mask = _prepare_4d_causal_attention_mask(
            attention_mask,
            (input_tensor.shape[0], input_tensor.shape[1]),
            input_tensor,
            0,
        )

    return attention_mask,position_ids

def get_causal_mask(input_tensor,attention_mask):
    dtype, device = input_tensor.dtype, input_tensor.device
    min_dtype = torch.finfo(dtype).min
    sequence_length = input_tensor.shape[1]
    target_length = sequence_length

    causal_mask = torch.full((sequence_length, target_length), fill_value=min_dtype, dtype=dtype, device=device)
    if sequence_length != 1:
        causal_mask = torch.triu(causal_mask, diagonal=1)

    cache_position = torch.arange(0, 0 + input_tensor.shape[1], device=device)
    position_ids = cache_position.unsqueeze(0)
    causal_mask *= torch.arange(target_length, device=device) > cache_position.reshape(-1, 1)
    causal_mask = causal_mask[None, None, :, :].expand(input_tensor.shape[0], 1, -1, -1)
    causal_mask = causal_mask.clone()  # copy to contiguous memory for in-place edit

    if attention_mask.dim() == 2:
        mask_length = attention_mask.shape[-1]
        padding_mask = causal_mask[..., :mask_length].eq(0.0) * attention_mask[:, None, None, :].eq(0.0)
        causal_mask[..., :mask_length] = causal_mask[..., :mask_length].masked_fill(padding_mask, min_dtype)
    elif attention_mask.dim() == 4:
        # backwards compatibility: we allow passing a 4D attention mask shorter than the input length with
        # cache. In that case, the 4D attention mask attends to the newest tokens only.
        if attention_mask.shape[-2] < cache_position[0] + sequence_length:
            offset = cache_position[0]
        else:
            offset = 0
        mask_shape = attention_mask.shape
        mask_slice = (attention_mask.eq(0.0)).to(dtype=dtype) * min_dtype
        causal_mask[
            : mask_shape[0], : mask_shape[1], offset : mask_shape[2] + offset, : mask_shape[3]
        ] = mask_slice

    #causal_mask = AttentionMaskConverter._unmask_unattended(causal_mask, min_dtype)
    causal_mask.mul(~torch.all(causal_mask == min_dtype, dim=-1, keepdim=True))
    return causal_mask,position_ids,cache_position
def get_llama_sys_que(sys,que):
    return f'<s>[INST] <<SYS>>\n{sys}\n<</SYS>>\n\n{que} [/INST]'

def get_llama_with_answer(que,ans):
    return f"""<s>[INST] {que} [/INST] {ans} </s>"""

def get_llama_without_answer(que):
    return f"""<s>[INST] {que} [/INST]"""

def get_llama_without_answer_cot(que):
    return f"""<s>[INST] Please provide a multi-hop explanation for the next question: {que} [/INST] """

def get_qwen_without_answer(que):
    return f"""<|im_start|>user\n{que}<|im_end|>\n<|im_start|>assistant\n"""

def get_qwen_without_answer_cot(que):
    return f"""<|im_start|>user\n Please provide a multi-hop explanation for the next question: {que}<|im_end|>\n<|im_start|>assistant\n"""

def get_vicuna_without_answer(que):
    return f"""USER: {que} ASSISTANT:"""

def get_list_llama_without_answer(que, cot):
    if cot == False:
        L = [get_llama_without_answer(line) for line in que]
    else:
        L = [get_llama_without_answer_cot(line) for line in que]
    return L
def get_list_qwen_without_answer(que, cot):
    if cot == False:
        L = [get_qwen_without_answer(line) for line in que]
    else:
        L = [get_qwen_without_answer_cot(line) for line in que]
    return L

def get_llama_without_answer_cot(que):
    return f"""<s>[INST] Please provide a multi-hop explanation for the next question : {que} [/INST] """
    
if __name__ == "__main__":
    
    set_seed()
    config = Config()

    with open(config.data_path, 'r', encoding='utf-8') as json_file:
        edit_data = json.load(json_file)
    with open(config.ex_data_path, 'r', encoding='utf-8') as json_file:
        ex_datas = json.load(json_file)
    ex_datas = [get_llama_without_answer(i['instruction']+i['input'])+i['output']  for i in ex_datas]
    
    model = AutoModelForCausalLM.from_pretrained(config.model_path, device_map=f"cuda:{config.device}")
    
    tok = AutoTokenizer.from_pretrained(config.model_path)
    tokenizer = AutoTokenizer.from_pretrained(config.model_path,padding_side='left')
    if config.model_name == 'LLama2-7B-Chat':
        tok.pad_token_id = tok.eos_token_id
        tokenizer.pad_token_id = tok.eos_token_id

    batch_size = config.batch_size
    num_batches = len(edit_data) // batch_size + (1 if len(edit_data) % batch_size else 0)
    edited_data = []
    for batch_index in tqdm(range(num_batches)):
        start_index = batch_index * batch_size
        end_index = start_index + batch_size
        batch = edit_data[start_index:end_index]
        for i in batch:
            i['question'] = get_llama_without_answer(i['question'])

            i['para_question'] = get_llama_without_answer(i['para_question'])
            if config.model_name == 'LLama2-7B-Chat':
                i['answer'] = i['answer']+'</s>'
            elif config.model_name == 'Qwen1.5-7B-Chat':
                i['answer'] = i['answer']+'<|im_end|>'

            i['sub_question'] = get_list_llama_without_answer(i['sub_question'], False)
        
        random_elements = random.sample(ex_datas, config.ex_data_num)
        
        weights_copy = execute_batch_unke(model, tok, config, batch,random_elements)

        

        # Edited Knowledge 
        for data in batch:
            question = tokenizer([data['question'],data['para_question']], return_tensors='pt', padding=True)
            #print(question.input_ids) 
            with torch.no_grad():
                generated_ids = model.generate(
                input_ids=question['input_ids'].to(f'cuda:{str(config.device)}'),
                attention_mask=question['attention_mask'].to(f'cuda:{str(config.device)}'),
                do_sample=True,
                temperature=0.001,
                max_new_tokens=512
            )
            generated_ids = [
                output_ids[len(input_ids):] for input_ids, output_ids in zip(question['input_ids'], generated_ids)
            ]
            output = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
            if batch_index < 10 // config.batch_size + 1:
                print(f"question:{data['question']}")
                print(output[0])
                print(f"question:{data['para_question']}")
                print(output[1])
            data['original_prediction'] = output[0]
            data['para_prediction'] = output[1]
            if config.model_name == 'LLama2-7B-Chat':
                data['answer'] = data['answer'][:-len('</s>')]
            elif config.model_name == 'Qwen1.5-7B-Chat':
                data['answer'] = data['answer'][:-len('<|im_end|>')]

        

        for data in batch:
            question = tokenizer(data['sub_question'], return_tensors='pt', padding=True)
            with torch.no_grad():
                generated_ids = model.generate(
                input_ids=question['input_ids'].to(f'cuda:{str(config.device)}'),
                attention_mask=question['attention_mask'].to(f'cuda:{str(config.device)}'),
                do_sample=True,
                temperature=0.001,
                max_new_tokens=512
            )

            generated_ids = [
                output_ids[len(input_ids):] for input_ids, output_ids in zip(question['input_ids'], generated_ids)
            ]

            output = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)

            if batch_index < 10 // config.batch_size + 1:
                print(f"question:{data['sub_question']}")
                print(output)

            data['sub_pred'] = output

        
        edited_data.extend(batch)
        if config.keep_original_weight:
            with torch.no_grad():
                for k, v in weights_copy.items():
                    nethook.get_parameter(model, k)[...] = v.to(f"cuda:{config.device}")
            
    path = '../output/result_bs1_ex20_step50_s_qwen2.json'
    with open(path, 'w', encoding='utf-8') as json_file: 
        json.dump(edited_data, json_file, ensure_ascii=False, indent=4)
    
    print(f"saving to {path}")
    




    

    



