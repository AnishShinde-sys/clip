from transformers import CLIPTokenizer

def text_preprocess(text, max_length=77):
    tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
    tokens = tokenizer(text, max_length=max_length, padding="max_length", truncation=True, return_tensors="pt")
    return tokens["input_ids"]

# Test with your text
text = "this is a dog"
tokens = text_preprocess(text)

print(f"Input text: '{text}'")
print(f"Token IDs shape: {tokens.shape}")
print(f"Token IDs: {tokens}")
print(f"\nFirst 10 tokens: {tokens[0][:10].tolist()}")

