from tqdm import tqdm
import numpy as np
import torch
import torch.nn.functional as F
from losses import log_prob_loss, log_perplexity
from utils import get_forbidden_toks, filter_forbidden_toks, get_unigram_probs 

def run_arca(args, model, tokenizer, embedding_table, output_str = None):
    # Fixed output is used in the reverse case
    fixed_output = output_str is not None
    run_metadata = {}
    args.batch_size = args.arca_batch_size
    embedding_dim = embedding_table.shape[1]
    # Avoid degenerate solutions + additional constraints specified in args
    forbidden_input_toks = get_forbidden_toks(args, tokenizer, n_total_toks = embedding_table.shape[0], 
            output = False, output_str = output_str)
    if not fixed_output:
        forbidden_output_toks = get_forbidden_toks(args, tokenizer, n_total_toks = embedding_table.shape[0], 
                output = True, output_str = output_str)
    # Whether or not to use a fixed prompt prefix
    use_pp = args.prompt_prefix is not None
    if use_pp:
        prefix_toks = torch.Tensor(tokenizer(args.prompt_prefix)['input_ids']).long().to(args.device)
        prefix_embeddings = embedding_table[prefix_toks].unsqueeze(0)
        prefix_embeddings = prefix_embeddings.repeat(args.batch_size, 1, 1).detach()
        prefix_length = prefix_embeddings.shape[1]

        mid_toks = np.array(tokenizer(args.prompt_mid)['input_ids'][1:])
        mid_length = len(mid_toks)

    vocab_size = embedding_table.shape[0]
    embedding_dim = embedding_table.shape[1]
    if fixed_output:
        output_toks = np.array(tokenizer(output_str)['input_ids'][1:])
        output_toks_tensor = torch.Tensor(tokenizer(output_str)['input_ids']).long().to('cuda')
        args.output_length = output_toks.shape[0]
        run_metadata['n_output_toks'] = args.output_length
        assert args.unigram_output_constraint is None

    curr_toks = np.random.choice(vocab_size, size = args.prompt_length + mid_length + args.output_length, replace = True)
    if fixed_output:
        curr_toks[args.prompt_length + mid_length:] = output_toks
        curr_toks[args.prompt_length: args.prompt_length + mid_length] = mid_toks
    if use_pp:
        curr_toks = np.concatenate([prefix_toks.detach().cpu().numpy(), curr_toks], axis = 0)
    stacked_cur_toks = np.tile(curr_toks, (args.batch_size, 1))
    curr_toks_tensor = torch.Tensor(stacked_cur_toks).long().to(args.device)
    
    if args.unigram_output_constraint is not None:
        output_unigram_lps = get_unigram_probs(args.unigram_output_constraint, gptj = args.model_id == 'gptj')
    if args.unigram_input_constraint is not None:
        input_unigram_lps = get_unigram_probs(args.unigram_input_constraint, gptj = args.model_id == 'gptj')

    output_start = args.prompt_length + prefix_length + mid_length if use_pp else args.prompt_length
    full_embeddings = torch.zeros(args.batch_size, args.prompt_length + args.output_length + mid_length, embedding_dim).to('cuda')
    # Initialize full embeddings
    for i in range(args.prompt_length + args.output_length + mid_length):
        rel_idx = i + prefix_length if use_pp else i
        full_embeddings[:, i] = embedding_table[curr_toks[rel_idx]].unsqueeze(0).repeat(args.batch_size, 1)
    # Iterate
    counter = 0
    best_toks = None
    best_score = -np.infty
    for it in tqdm(range(args.arca_iters)):
        for tok_id in range(args.prompt_length):
            counter += 1
            tok_in_output = tok_id >= args.prompt_length
            # Output tokens remain fixed in the reversing case
            if tok_in_output and fixed_output:
                continue
            update_idx = tok_id + prefix_length if use_pp else tok_id
            new_indices = np.random.choice(vocab_size, size = args.batch_size, replace = True)
            if args.autoprompt:
                new_indices = curr_toks[update_idx].repeat(args.batch_size)
            tok_embeddings = embedding_table[new_indices, :] 
            full_embeddings[:, tok_id, :] = embedding_table[new_indices, :] 
            if args.model_id == 'gptj' or args.model_id == "llama2":
                full_embeddings = full_embeddings.half()
            # Update to compute the perplexity loss
            stacked_cur_toks[:, update_idx] = new_indices
            curr_toks_tensor[:, update_idx] = torch.Tensor(new_indices).long().to('cuda')
            if use_pp:
                labels = torch.cat([-100 * torch.ones(args.prompt_length + prefix_length + mid_length).to('cuda').unsqueeze(0).repeat(args.batch_size, 1), curr_toks_tensor[:, args.prompt_length + prefix_length + mid_length:]], dim = 1).long()
            else:
                labels = torch.cat([-100 * torch.ones(args.prompt_length).to('cuda').unsqueeze(0).repeat(args.batch_size, 1), curr_toks_tensor[:, args.prompt_length:]], dim = 1).long()
            #full_embeddings = full_embeddings.detach()
            #if full_embeddings.requires_grad:
            #    full_embeddings.grad.zero_()
            #full_embeddings.requires_grad = True
            #full_embeddings.retain_grad()
            tok_embeddings = tok_embeddings.unsqueeze(dim=1)
            tok_embeddings.requires_grad_()
            if use_pp:
                out = model(inputs_embeds = torch.cat([prefix_embeddings, full_embeddings[:, : tok_id, :], tok_embeddings, full_embeddings[:, tok_id+1:, :]], dim = 1), labels = labels)
            else:
                out = model(inputs_embeds = full_embeddings, labels = labels)
            loss = log_prob_loss(out, labels, temp = 1)
            # Comptue the perplexity loss
            if args.lam_perp > 0:
                perp_loss = log_perplexity(out, stacked_cur_toks[:,:output_start])
                loss += args.lam_perp * perp_loss
            loss.backward()
            grad = tok_embeddings.grad
            tok_embeddings.requires_grad_(False)
            backward_scores = - torch.matmul(embedding_table, grad.squeeze(dim=1).mean(dim = 0))
            if tok_in_output and not args.autoprompt:
                forward_log_probs = F.log_softmax(out.logits[0, update_idx - 1, :], dim = 0)
                scores = backward_scores + forward_log_probs
                if args.unigram_output_constraint is not None:
                    scores += args.unigram_weight * output_unigram_lps
            else:
                scores = backward_scores
                if args.unigram_input_constraint is not None:
                    scores += args.unigram_weight * input_unigram_lps
                    
            best_scores_idxs = scores.argsort(descending = True)
            if tok_in_output:
                best_scores_idxs = filter_forbidden_toks(best_scores_idxs, forbidden_output_toks)
            else:
                best_scores_idxs = filter_forbidden_toks(best_scores_idxs, forbidden_input_toks)
            full_embeddings= full_embeddings.detach()
            with torch.no_grad():
                full_embeddings[:, tok_id, :] = embedding_table[best_scores_idxs[:args.batch_size], :]                
                stacked_cur_toks[:, update_idx] = best_scores_idxs[:args.batch_size].cpu().detach().numpy()
                curr_toks_tensor[:, update_idx] = best_scores_idxs[:args.batch_size]
                if use_pp:
                    out = model(inputs_embeds = torch.cat([prefix_embeddings, full_embeddings], dim = 1))
                else:
                    out = model(inputs_embeds = full_embeddings)
                log_probs = F.log_softmax(out.logits[:, -1 - args.output_length: -1, :], dim = 2)
                batch_log_probs = torch.stack([log_probs[i, torch.arange(args.output_length), curr_toks_tensor[i, output_start:]].sum() for i in range(args.batch_size)])
                if args.lam_perp > 0:
                    output_perps = log_perplexity(out, stacked_cur_toks[:,:output_start], ret_all = True)
                    batch_log_probs -= args.lam_perp * output_perps
                if args.unigram_output_constraint is not None and tok_in_output:
                    batch_log_probs += args.unigram_weight * output_unigram_lps[best_scores_idxs[:args.batch_size]]
                elif args.unigram_input_constraint is not None and not tok_in_output:
                    batch_log_probs += args.unigram_weight * input_unigram_lps[best_scores_idxs[:args.batch_size]]
                best_batch_idx = batch_log_probs.argmax()
                best_idx = best_scores_idxs[best_batch_idx]
                curr_toks[update_idx] = best_idx.item()
                stacked_cur_toks[:, update_idx] = best_idx.item()
                curr_toks_tensor[:, update_idx] = best_idx.item()
                full_embeddings[:, tok_id, :] = embedding_table[best_idx].unsqueeze(0).repeat(args.batch_size, 1)
                gen_output = log_probs[best_batch_idx].argmax(dim = 1)
                actual_output = curr_toks_tensor[0][output_start:]
                # Can modify success conditions here to stop running the algorithm
                if (tok_id == args.prompt_length - 1):
                    print(tokenizer.decode(gen_output))
                if (batch_log_probs[best_batch_idx] > best_score):
                    best_score = batch_log_probs[best_batch_idx]
                    best_toks = curr_toks.copy()
                output_matches = (actual_output == gen_output).all().item()
                if args.unigram_input_constraint is not None:
                    input_unigram_satisfied  = torch.exp(input_unigram_lps[curr_toks[:output_start]].min()).item() > 0.99
                else:
                    input_unigram_satisfied = True
                if args.unigram_output_constraint is not None and not fixed_output:
                    output_unigram_satisfied = torch.exp(output_unigram_lps[curr_toks[output_start:]].max()).item() > 0.5
                else:
                    output_unigram_satisfied = True
                # Success condition
                if counter == args.arca_iters * args.prompt_length and input_unigram_satisfied and output_unigram_satisfied:
                    if args.lam_perp > 0:
                        run_metadata['perplexity'] = output_perps[best_batch_idx].item()
                    if args.unigram_output_constraint is not None:
                        run_metadata['output_unigram'] = torch.exp(output_unigram_lps[curr_toks[output_start:]]).mean().item()
                        run_metadata['max_output_unigram'] = torch.exp(output_unigram_lps[curr_toks[output_start:]].max()).item()
                        run_metadata['min_output_unigram'] = torch.exp(output_unigram_lps[curr_toks[output_start:]].min()).item()
                    if args.unigram_input_constraint is not None:
                        run_metadata['input_unigram'] = torch.exp(input_unigram_lps[curr_toks[:output_start]]).mean().item()
                        run_metadata['max_input_unigram'] = torch.exp(input_unigram_lps[curr_toks[:output_start]].max()).item()
                        run_metadata['min_input_unigram'] = torch.exp(input_unigram_lps[curr_toks[:output_start]].min()).item()
                    if fixed_output:
                        best_toks = best_toks[:-args.output_length]
                    return best_toks, it, run_metadata
    # Failure case
    if args.lam_perp > 0:
        run_metadata['perplexity'] = None
        if args.unigram_output_constraint is not None:
            run_metadata['output_unigram'] = -1
        elif args.unigram_input_constraint is not None:
            run_metadata['input_unigram'] = -1
    return -1, -1, run_metadata
