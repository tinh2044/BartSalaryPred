import torch
import numpy as np
import json
from transformers import BartphoTokenizer

class Tokenizer:
    def __init__(self, repo_id, prune_id_file):
        self.tokenizer = BartphoTokenizer.from_pretrained(repo_id)
        self.prune_id_file = prune_id_file
        self.pad_index = self.tokenizer.convert_tokens_to_ids('<pad>')
        self.ignore_index = self.pad_index
        
        with open(self.prune_id_file, 'r') as f:
            self.pruneids = json.load(f)
            self.pruneids = {int(k): v for k, v in self.pruneids.items()}
            for t in ['<pad>', '<s>', '</s>', '<unk>']:
              id_ = self.tokenizer.convert_tokens_to_ids(t)
              assert self.pruneids[id_] == id_, '{}->{}'.format(id_, self.pruneids[id_])
        
        self.pruneids_reverse = {i2: i1 for i1, i2 in self.pruneids.items()}
        self.sos_index = self.pruneids[self.tokenizer.convert_tokens_to_ids('<s>')]
        self.eos_index = self.pruneids[self.tokenizer.convert_tokens_to_ids('</s>')]

    def __len__(self): 
        return len(self.pruneids)
        
    def prune(self, input_ids):
        pruned_input_ids = []
        for single_seq in input_ids:
            pruned_single_seq = []
            for id_ in single_seq:
                if not id_ in self.pruneids:
                    new_id = self.pruneids[self.tokenizer.convert_tokens_to_ids('<unk>')]
                else:
                    new_id = self.pruneids[id_]
                pruned_single_seq.append(new_id)
            pruned_input_ids.append(pruned_single_seq)
        return torch.tensor(pruned_input_ids, dtype=torch.long)

    def encode_tgt(self, input_str, max_lenght=None):
        with self.tokenizer.as_target_tokenizer():
            if max_lenght is None:
                raw_outputs = self.tokenizer(input_str, return_attention_mask=True,
                                             return_length=True,
                                             padding='longest')
            else:
                raw_outputs = self.tokenizer(input_str, return_attention_mask=True,
                                             return_length=True, max_length=max_lenght, 
                                             padding='max_length')

            input_ids = self.prune(raw_outputs['input_ids'])

        return {
            'labels': input_ids
        }

    def encode_src(self, input_str):
        raw_outputs = self.tokenizer(input_str, padding="longest", return_attention_mask=True)

        raw_outputs['input_ids'] = self.prune(raw_outputs['input_ids'])
        raw_outputs['attention_mask'] = torch.from_numpy(np.array(raw_outputs['attention_mask'])).long()

        return raw_outputs

    def shift_tokens_right(self, input_ids, pad_token_id, ignore_index= -100):
        prev_output_tokens = input_ids.clone()

        assert pad_token_id is not None, "self.model.config.pad_token_id has to be defined."
        prev_output_tokens.masked_fill_(prev_output_tokens == -100, pad_token_id)
        index_of_eos = (prev_output_tokens.ne(pad_token_id).sum(dim=1) - 1).unsqueeze(-1)
        for ii, ind in enumerate(index_of_eos.squeeze(-1)):
            input_ids[ii, ind:] = ignore_index
        decoder_start_tokens = prev_output_tokens.gather(1, index_of_eos).squeeze()
        prev_output_tokens[:, 1:] = prev_output_tokens[:, :-1].clone()
        prev_output_tokens[:, 0] = decoder_start_tokens

        return prev_output_tokens

    def prune_reverse(self, pruned_input_ids):
        batch_size, max_len = pruned_input_ids.shape
        input_ids = pruned_input_ids.clone()
        for b in range(batch_size):
            for i in range(max_len):
                id_ = input_ids[b, i].item()
                if not id_ in self.pruneids_reverse:
                    new_id = self.tokenizer.convert_tokens_to_ids('<unk>')
                else:
                    new_id = self.pruneids_reverse[id_]
                input_ids[b, i] = new_id
        return input_ids

    def batch_decode(self, sequences):
        sequences = sequences[:, 1:]
        sequences_ = self.prune_reverse(sequences)
        decoded_sequences = self.tokenizer.batch_decode(sequences_, skip_special_tokens=True)
        for di, d in enumerate(decoded_sequences):
              if len(d) > 2 and d[-1] == '.' and d[-2] != ' ':
                  d = d[:-1] + ' .'
                  decoded_sequences[di] = d
        return decoded_sequences 