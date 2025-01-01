from transformers import GPT2LMHeadModel, GPT2Tokenizer

model = "gpt2-xl"
# model = "gpt2-large" model can be changed here

tokenizer = GPT2Tokenizer.from_pretrained(model)
model = GPT2LMHeadModel.from_pretrained(model)

model.eval()

model.generation_config.pad_token_id = tokenizer.pad_token_id


def generate_text(prompt, max_length=60, temperature=0.9):

    input_ids = tokenizer.encode(prompt, return_tensors="pt")
    output = model.generate(
        input_ids,
        max_length=max_length,
        num_return_sequences=1,
        no_repeat_ngram_size=2,
        top_k=25,
        top_p=0.90,
        temperature=temperature,
    )

    generated_text = tokenizer.decode(output[0])
    return generated_text

def generate(prompt):
  generated_text = generate_text(prompt, max_length = 60)
  print(generated_text)

# Testing
generate('What are winds?')