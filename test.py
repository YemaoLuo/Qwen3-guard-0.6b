import time

from transformers import AutoModelForCausalLM, AutoTokenizer

# Load model and tokenizer from local directory
local_model_path = "./guard-0.6b"
print("Loading model and tokenizer from local directory...")
tokenizer = AutoTokenizer.from_pretrained(local_model_path)
model = AutoModelForCausalLM.from_pretrained(local_model_path)
print("Model loaded successfully!\n")

# Interactive loop
while True:
    user_input = input("Enter your message (or 'quit' to exit): ").strip()
    
    if user_input.lower() == 'quit':
        print("Exiting...")
        break
    
    if not user_input:
        print("Please enter a valid message.\n")
        continue
    
    # Prepare messages
    messages = [
        {"role": "user", "content": user_input},
    ]
    
    # Start timing
    start_time = time.time()
    
    # Tokenize and generate
    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    ).to(model.device)
    
    outputs = model.generate(**inputs, max_new_tokens=40)
    
    # End timing
    end_time = time.time()
    time_cost = end_time - start_time
    
    # Decode and print response
    response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[-1]:])
    
    print(f"\nModel Response: {response}")
    print(f"Time Cost: {time_cost:.4f} seconds\n")
