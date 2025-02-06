def generate_text(tokenizer, inputs, model, max_length=50, temperature=0.7, top_k=50):
    output = model.generate(
        **inputs, 
        max_length=max_length, 
        do_sample=True, 
        temperature=temperature, 
        top_k=top_k
    )
    
    return tokenizer.decode(output[0], skip_special_tokens=True)