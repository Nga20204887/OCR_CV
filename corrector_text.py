
# from utils.vietnamese_normalizer import clean_text
# from transformers import pipeline

# corrector_model = pipeline("text2text-generation", model="bmd1905/vietnamese-correction")

# def corrector(text):
#     text = text.lower()
#     corrector_text = clean_text(corrector_model(text)[0]['generated_text'])
#     return corrector_text

# # text = 'NGÔ GIAHUY'
# # print(corrector(text))