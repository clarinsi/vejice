from transformers import BertForMaskedLM, BertModel, BertTokenizer

import torch

###---If you get error with initialization of libiomp5.dylib, try uncommenting these two lines
#import os
#os.environ['KMP_DUPLICATE_LIB_OK']='True'
###

tokenizer = BertTokenizer.from_pretrained('bert_model_finetuned/', do_lower_case=True)
finetuned_model_dir = 'bert_model_finetuned/'

model = BertForMaskedLM.from_pretrained(finetuned_model_dir)

#Model has to be put into evaluation mode before we can use it
model.eval()

sentence = "Dobra stran arhitekture BERT je da jo lahko samo izpopolnimo in tako ohranimo že vsebovano jezikovno znanje ."

masked_sentence = """[CLS] Dobra [MASK] stran [MASK] arhitekture [MASK] bert [MASK] je [MASK] da [MASK] jo [MASK] lahko [MASK] samo [MASK] izpopolnimo [MASK] in [MASK] tako [MASK] ohranimo [MASK] že [MASK] vsebovano [MASK] jezikovno [MASK] znanje [MASK] . [SEP]"""

input_ids = []
attention_masks = []
label_list = []

encoded_dict = tokenizer.encode_plus(
                    masked_sentence,
                    add_special_tokens = False,
                    max_length = 128,
                    padding = 'max_length',
                    truncation = True,
                    return_attention_mask = True,
                    return_tensors = 'pt',
                )

input_ids.append(encoded_dict['input_ids'])
attention_masks.append(encoded_dict['attention_mask'])

input_ids = torch.cat(input_ids, dim=0)
attention_masks = torch.cat(attention_masks, dim=0)

predictions , true_labels = [], []

outputs = model(input_ids, attention_mask=attention_masks)
logits = outputs[0]
input_ids = input_ids[0].numpy()

predicted_sentence = ""
index_mask = 0
origin_counter_sentence = 0

for token in masked_sentence.split(" "):
    if token != "[CLS]" and token != "[SEP]":
        if token == "[MASK]":
            while input_ids[index_mask] != 105:
                index_mask = index_mask + 1
            if torch.argmax(logits[0, index_mask]).item() == 1:
                predicted_sentence += ","
            index_mask += 1
        else:
            if index_mask > 0 and token != ".":
                predicted_sentence += " "
            predicted_sentence += sentence.split(' ')[origin_counter_sentence]
            origin_counter_sentence += 1

print(predicted_sentence)


