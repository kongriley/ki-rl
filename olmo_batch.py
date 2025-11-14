import torch

def generate_text_batch(batch_messages, model, tokenizer, max_new_tokens=256, temperature=1, top_p=0.9):
    """Generate text for a batch of messages using Hugging Face model."""
    input_texts = []
    for messages in batch_messages:
        if hasattr(tokenizer, 'apply_chat_template'):
            input_text = tokenizer.apply_chat_template(
                messages, 
                tokenize=False, 
                add_generation_prompt=True
            )
        else:
            input_text = messages[0]["content"]
        input_texts.append(input_text)
    
    inputs = tokenizer(
        input_texts, 
        return_tensors="pt", 
        padding=True,
        truncation=True,
        max_length=4096
    ).to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id
        )
    
    generated_texts = []
    for i, output in enumerate(outputs):
        input_length = inputs.input_ids[i].shape[0]
        generated_text = tokenizer.decode(
            output[input_length:], 
            skip_special_tokens=True
        )
        generated_texts.append(generated_text)
    
    return generated_texts

def pipeline(data, model, tokenizer): 
    return generate_text_batch(data, model, tokenizer)