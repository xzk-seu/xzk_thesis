from transformers import AutoTokenizer, AutoModel, AutoModelForTokenClassification
import torch

tokenizer = AutoTokenizer.from_pretrained("./BERTOverflow")

model = AutoModel.from_pretrained("./BERTOverflow")

sequence = "Hugging Face Inc. is a company based in New York City. Its headquarters are in DUMBO, therefore very" \
            "close to the Manhattan Bridge."


tokens = tokenizer.tokenize(tokenizer.decode(tokenizer.encode(sequence)))
inputs = tokenizer.encode(sequence, return_tensors="pt")

m = model(inputs)
outputs = model(inputs).logits
predictions = torch.argmax(outputs, dim=2)


print([(token, label_list[prediction]) for token, prediction in zip(tokens, predictions[0].numpy())])

