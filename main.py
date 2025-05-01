from llama_cpp import Llama
from personality import apply_personality
from context_manager import ContextManager
from audio import playnow

# Load GGUF model with 33 layers on GPU
llm = Llama(
    model_path="./your_model_file.gguf",
    n_gpu_layers=33,
    use_mlock=True,
    verbose=False,
    n_ctx=8192
)

ctx_mgr = ContextManager()

def generate_response(user_input: str) -> str:
    # Build prompt with context
    context = ctx_mgr.get_context()
    full_prompt = f"{context}\nUser: {user_input}\nBot:"
    prompt_with_personality = apply_personality(full_prompt)

    output = llm(prompt_with_personality, stop=["User:", "Bot:"], max_tokens=200)
    response = output["choices"][0]["text"].strip()
    
    ctx_mgr.update(user_input, response)
    return response

if __name__ == "__main__":
    print("Sage: Hello! What's up?")
    while True:
        user_input = input("You: ")
        if user_input.lower() in {"quit", "exit"}:
            break
        response = generate_response(user_input)
        playnow(response)
        print(f"Sage: {response}")
