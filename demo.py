from transformers import CamembertTokenizer, CamembertForMaskedLM

import torch

###---If you get error with initialization of libiomp5.dylib, try uncommenting these two lines
#import os
#os.environ['KMP_DUPLICATE_LIB_OK']='True'
###

tokenizer = CamembertTokenizer.from_pretrained('sloBERTa_model_finetuned/', do_lower_case=True)
finetuned_model_dir = 'sloBERTa_model_finetuned/'
mask_token_id = 32004

model = CamembertForMaskedLM.from_pretrained(finetuned_model_dir)

#Model has to be put into evaluation mode before we can use it
model.eval()

sentence = "Dobra stran arhitekture BERT je da jo lahko samo izpopolnimo in tako ohranimo že vsebovano jezikovno znanje ."

masked_sentence = """<s> Dobra <mask> stran <mask> arhitekture <mask> bert <mask> je <mask> da <mask> jo <mask> lahko <mask> samo <mask> izpopolnimo <mask> in <mask> tako <mask> ohranimo <mask> že <mask> vsebovano <mask> jezikovno <mask> znanje <mask> . </s>"""

input_ids = []
attention_masks = []
label_list = []

encoded_dict = tokenizer.encode_plus(
                    masked_sentence,
                    add_special_tokens = False,
                    max_length = 200,
                    #pad_to_max_length = True,
                    padding='max_length',
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
    if token != "<s>" and token != "</s>":
        if token == "<mask>":
            while input_ids[index_mask] != mask_token_id:
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


