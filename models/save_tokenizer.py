from transformers import AutoTokenizer


model_directory = './toxic_comment_model'  



tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')


tokenizer.save_pretrained('./toxic_comment_model')

print("Tokenizer saved successfully!")

