from transformers import pipeline, set_seed

generator = pipeline("text-generation", model="gpt2")
set_seed(42)

prompt = "Artificial Intelligence will transform the future of"
outputs = generator(
    prompt,
    max_length=60,
    num_return_sequences=2,
    temperature=0.8,
    top_k=50,
    top_p=0.95,
    do_sample=True
)

for i, out in enumerate(outputs, 1):
    print(f"--- Generated Text {i} ---")
    print(out["generated_text"])
    print()
