import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2")

def generate_paragraph(prompt_text, max_length=200, num_return_sequences=1, temperature=0.7, top_k=50, top_p=0.95, repetition_penalty=1.0):
    input_ids = tokenizer.encode(prompt_text, return_tensors='pt')
    attention_mask = torch.ones(input_ids.shape, dtype=torch.long)
    
    output = model.generate(
        input_ids,
        attention_mask=attention_mask,
        max_length=max_length,
        num_return_sequences=num_return_sequences,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        repetition_penalty=repetition_penalty,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id
    )
    
    generated_texts = [tokenizer.decode(output_ids, skip_special_tokens=True) for output_ids in output]
    return generated_texts

# Example 1: Generate text about "The Future of AI"
prompt1 = "The future of AI is incredibly promising, with advancements that could revolutionize various industries."
generated_texts_1 = generate_paragraph(prompt1, max_length=250, num_return_sequences=1)
print("--- Generated Text on The Future of AI ---")
print(generated_texts_1[0])
print("\n")

# Example 2: Generate text about "Sustainable Living Practices"
prompt2 = "Sustainable living practices are essential for preserving our planet's resources for future generations."
generated_texts_2 = generate_paragraph(prompt2, max_length=250, num_return_sequences=1)
print("--- Generated Text on Sustainable Living Practices ---")
print(generated_texts_2[0])
print("\n")

# Example 3: Generate text about "The Benefits of Reading"
prompt3 = "Reading offers numerous benefits, from expanding knowledge to improving cognitive functions."
generated_texts_3 = generate_paragraph(prompt3, max_length=250, num_return_sequences=1)
print("--- Generated Text on The Benefits of Reading ---")
print(generated_texts_3[0])
print("\n")

# Example 4: Generate text about "Space Exploration"
prompt4 = "Space exploration continues to captivate humanity, pushing the boundaries of our understanding of the universe."
generated_texts_4 = generate_paragraph(prompt4, max_length=250, num_return_sequences=1)
print("--- Generated Text on Space Exploration ---")
print(generated_texts_4[0])
print("\n")

# Example 5: Generate text about "The Impact of Social Media"
prompt5 = "Social media has profoundly impacted modern society, connecting people globally but also raising concerns about privacy and mental health."
generated_texts_5 = generate_paragraph(prompt5, max_length=250, num_return_sequences=1)
print("--- Generated Text on The Impact of Social Media ---")
print(generated_texts_5[0])
print("\n")
