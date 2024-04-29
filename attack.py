import time
import numpy as np
import torch
import torch.nn as nn
import concurrent.futures
from fastchat.conversation import SeparatorStyle

def concat_user_ids(begin_ids, prompt_ids, middle_ids, prompt_replace_lst = None):
    user_ids_lst = []
    user_ids = begin_ids.copy()
    replace_slice_lst=[]
    prompt_slice = slice(len(user_ids), -1)
    last_slice_stop = len(user_ids)
    for id in prompt_ids:
        user_ids += [id]
        replace_slice_lst.append(slice(last_slice_stop, len(user_ids)))
        last_slice_stop = len(user_ids)
    prompt_slice = slice(prompt_slice.start, len(user_ids))
    user_ids += middle_ids

    if (prompt_replace_lst == None):
        return [user_ids]

    for i, repl_lst in enumerate(prompt_replace_lst):
        for repl_token in repl_lst:
            cand_user_ids = user_ids.copy()
            cand_user_ids[replace_slice_lst[i]] = [repl_token]
            user_ids_lst.append(cand_user_ids)

    return user_ids_lst

def concat_full_ids(user_ids_lst, target_ids):
    input_ids_lst = []
    loss_slice = slice(len(user_ids_lst[0]) - 1, len(user_ids_lst[0]) + len(target_ids) - 1)
    for user_ids in user_ids_lst:
        input_ids = user_ids + target_ids
        input_ids_lst.append(input_ids)
    return input_ids_lst, loss_slice


def get_loss(model, batch_ids_lst, target_ids, device):
    input_ids_lst, loss_slice = concat_full_ids(batch_ids_lst, target_ids)
    input_ids_lst = torch.tensor(input_ids_lst, device=device)
    attention_mask = torch.ones_like(input_ids_lst).long()
    position_ids = attention_mask.cumsum(-1) - 1
    logits = model(input_ids=input_ids_lst, attention_mask=attention_mask, position_ids=position_ids).logits
    target = torch.tensor(target_ids, device=device).unsqueeze(0).repeat((logits.shape[0], 1))
    batch_losses = nn.CrossEntropyLoss(reduction='none')(logits[:,loss_slice,:].transpose(1, 2), target).sum(dim=-1)

    return batch_losses

def get_loss_from_ids(models, executor, system_ids, prompt_ids, prompt_replace_lst = None):
    cuda_num = torch.cuda.device_count()
    user_ids_lst = concat_user_ids(system_ids["begin_ids"], prompt_ids, system_ids["middle_ids"], prompt_replace_lst)
    losses = torch.empty((0), dtype=torch.bfloat16, device=models[0].device)
    batch_size = 64
    for i in range(int(np.ceil(len(user_ids_lst) / batch_size))):
        batch_num = np.min([len(user_ids_lst) - i*batch_size, batch_size])
        if (batch_num <= 0):
            break
        batch_ids_lst = user_ids_lst[i*batch_size: i*batch_size + batch_num]
    
        if (cuda_num == 1):
            loss = get_loss(models[0], batch_ids_lst, system_ids["target_ids"], "cuda:0")
            losses = torch.cat((losses, loss))
        else:
            future_lst = []
            thread_size = int(np.ceil(batch_size / cuda_num))
            for j in range(cuda_num):
                thread_num = np.min([len(batch_ids_lst) - j*thread_size, thread_size])
                if (thread_num <= 0):
                    break
                thread_ids_lst = batch_ids_lst[j*thread_size: j*thread_size + thread_num]
                future = executor.submit(get_loss, models[j], thread_ids_lst, system_ids["target_ids"], f"cuda:{j}")
                future_lst.append(future)

            for future in future_lst:
                losses = torch.cat((losses, future.result().to(models[0].device)))

    return losses


def select_prompt(models, tokenizer, executor,system_ids_lst, prompt_ids, prompt_replace_lst, filter_word, previous_loss):  

    filtered_lst = prompt_replace_lst.tolist().copy()

    for i, repl_lst in enumerate(prompt_replace_lst):
        for repl_token in repl_lst:
            cand_prompt_ids = prompt_ids.tolist()
            cand_prompt_ids[i] = repl_token
            decoded_str = tokenizer.decode(cand_prompt_ids)
            if (cand_prompt_ids != tokenizer.encode(decoded_str)[2:] or filter_word in decoded_str.lower()):
                filtered_lst[i].remove(repl_token)

    index_lst = [(i, j) for i, sublst in enumerate(filtered_lst) for j, _ in enumerate(sublst)]
    losses_all = torch.zeros(len(index_lst), dtype=torch.bfloat16, device=models[0].device)
    for system_ids in system_ids_lst:
        losses_all += get_loss_from_ids(models, executor, system_ids, prompt_ids, filtered_lst)

    sorted_indices = losses_all.argsort()
    replace_pos = []
    new_prompt = None
    new_loss = previous_loss
    for indice in sorted_indices:
        best_index = indice
        prompt_index = index_lst[best_index][0]
        repl_index = index_lst[best_index][1]
        if (prompt_index in replace_pos):
            continue
        if (prompt_ids[prompt_index] != filtered_lst[prompt_index][repl_index]):
            replace_pos.append(prompt_index)
            #old_id = prompt_ids[prompt_index].clone()
            prompt_ids[prompt_index] = filtered_lst[prompt_index][repl_index]
            loss = 0
            for system_ids in system_ids_lst:
                loss += get_loss_from_ids(models, executor, system_ids, prompt_ids).item()

            if ((loss < new_loss or new_prompt == None) and (prompt_ids.tolist() == tokenizer.encode(tokenizer.decode(prompt_ids))[2:])):
                #print(f"{loss=} {tokenizer.decode(old_id)} -> {tokenizer.decode(prompt_ids[prompt_index])}")
                new_prompt = prompt_ids.clone()
                new_loss = loss
            else:
                break

    assert new_prompt != None
    return new_prompt, new_loss


def cal_replacable_ids(model, system_ids_lst, prompt_ids, repl_ids):
    top_k = 128
    select_b = 16

    embed_weights = model.model.embed_tokens.weight

    one_hot = torch.zeros(prompt_ids.shape[0],embed_weights.shape[0],
                          device=model.device,dtype=embed_weights.dtype)
    
    one_hot.scatter_(1, prompt_ids.unsqueeze(1),
                     torch.ones(one_hot.shape[0], 1, device=model.device, dtype=embed_weights.dtype))
    
    one_hot.requires_grad_()
    
    for system_ids in system_ids_lst:
        input_embeds = (one_hot @ embed_weights).unsqueeze(0)

        user_embeds = torch.cat([system_ids["begin_embeds"], input_embeds,system_ids["middle_embeds"]], dim=1)
        
        full_embeds = torch.cat([user_embeds, system_ids["target_embeds"]], dim=1)
        loss_slice = slice(len(user_embeds[0])-1, len(full_embeds[0])-1)
        
        logits = model(inputs_embeds=full_embeds).logits
        loss = nn.CrossEntropyLoss()(logits[0,loss_slice,:], torch.tensor(system_ids["target_ids"], device=model.device))
        
        loss.backward()
    
    grad = one_hot.grad.clone()
    one_hot.requires_grad_(False)
    
    grad = grad / grad.norm(dim=-1, keepdim=True)
    
    ref_grad = torch.full_like(grad, -np.infty)
    ref_grad[:, repl_ids] = -grad[:, repl_ids]

    prompt_replace_lst = ref_grad.topk(top_k, dim=1).indices[:,torch.randperm(top_k, device=model.device)[:select_b]]
    return prompt_replace_lst


def assemble_ids(tokenizer, template, question, target):  
    def find_ids(fst_ids, sed_ids):
        length = len(sed_ids)
        for index in range(len(input_ids) - length + 1):
            if fst_ids[index:index+length] == sed_ids:
                return index
        raise ValueError("Cannot find ids")
    
    template.messages = []

    template.append_message(template.roles[0], "<unk>")
    input_ids = tokenizer.encode(template.get_prompt())
    begin_ids = input_ids[0: input_ids.index(tokenizer.unk_token_id)]

    if (template.sep_style == SeparatorStyle.CHATML):
        target_ids = tokenizer.encode("\n" + target)[3:]
    else:
        target_ids = tokenizer.encode(target)[1:]

    template.update_last_message(question)
    template.append_message(template.roles[1], target)
    input_ids = tokenizer.encode(template.get_prompt())
    middle_ids = input_ids[len(begin_ids): find_ids(input_ids, target_ids)]

    return begin_ids, middle_ids, target_ids


def get_replacable_ids(tokenizer, filter_word, device):
    repl_ids = []
    repl_tokens = tokenizer.convert_ids_to_tokens(range(0, tokenizer.vocab_size))
    for i, token in enumerate(repl_tokens):
        if token.isascii() and token.isalpha() and filter_word not in token.lower():
            repl_ids.append(i)

    return torch.tensor(repl_ids, device=device)


def generate_output(model, tokenizer, input_ids, max_new_tokens):
    gen_config = model.generation_config
    
    gen_config.max_new_tokens = max_new_tokens
    gen_config.do_sample = False
    gen_config.temperature = 1.0
    gen_config.top_p = 1.0
    gen_config.pad_token_id = tokenizer.pad_token_id

    output_ids = model.generate(input_ids.unsqueeze(0), generation_config=gen_config)[0]

    return output_ids[len(input_ids):]


def generate_suffix(models, tokenizer, template_lst, question, target, num_epoch, token_nums, filter_word, seed = 0):
    for model in models:    
        model.eval()
        model.requires_grad_(False)

    model = models[0]

    executor = concurrent.futures.ThreadPoolExecutor(max_workers=torch.cuda.device_count())

    torch.manual_seed(seed)

    repl_ids = get_replacable_ids(tokenizer, filter_word, device=model.device)

    while True:
        prompt_ids = repl_ids[torch.randperm(len(repl_ids), device=model.device)[:token_nums]]
        decoded_str = tokenizer.decode(prompt_ids)
        if (prompt_ids.tolist() == tokenizer.encode(decoded_str)[2:] and 
            filter_word not in decoded_str):
            break

    previous_loss = np.infty
    best_loss = np.infty
    best_prompt_id = prompt_ids.clone()
    system_ids_lst = []
    for template in template_lst:
        begin_ids, middle_ids, target_ids = assemble_ids(tokenizer, template, question, target)
        begin_embeds = model.model.embed_tokens(torch.tensor(begin_ids, device=model.device).unsqueeze(0))
        middle_embeds = model.model.embed_tokens(torch.tensor(middle_ids, device=model.device).unsqueeze(0))
        target_embeds = model.model.embed_tokens(torch.tensor(target_ids, device=model.device).unsqueeze(0))
        system_ids_lst.append({"begin_ids":begin_ids, 
                               "middle_ids":middle_ids,
                               "target_ids":target_ids,
                               "begin_embeds":begin_embeds,
                               "middle_embeds":middle_embeds,
                               "target_embeds":target_embeds})

    for it in range(num_epoch):

        cal_time = time.perf_counter()

        prompt_replace_lst = cal_replacable_ids(model, system_ids_lst, prompt_ids, repl_ids)

        prompt_ids, loss = select_prompt(models, tokenizer,executor, system_ids_lst, prompt_ids, prompt_replace_lst, filter_word, previous_loss)

        if (loss < best_loss):
            best_loss = loss
            best_prompt_id = prompt_ids.clone()

        print(f"{it} Loss: {previous_loss} -> {loss} Best: {best_loss} Cal: {round(time.perf_counter()-cal_time, 2)}s")
        previous_loss = loss

        for system_ids in system_ids_lst:
            user_ids_lst = concat_user_ids(system_ids["begin_ids"], prompt_ids, system_ids["middle_ids"])
            user_ids = torch.tensor(user_ids_lst[0], device=model.device)

            gen_str = tokenizer.decode(generate_output(model, tokenizer, user_ids, 8))
            print(f"Answer: |{gen_str}|")

    return tokenizer.decode(best_prompt_id)

