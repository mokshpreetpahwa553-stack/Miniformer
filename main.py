import numpy as np

from transformer import build_vocabulary, tokenizer, token_to_id
from transformer_layer import transformer


def main():
    print("MiniFormer — type text; Type 'quit' to exit.\n")

    print("Building vocabulary from Wikipedia...")
    vocab = build_vocabulary()


    d_model = 64
    E = np.random.rand(len(vocab) + 1, d_model) * 0.1

    # 3. Loop: read input, encode, print vector
    while True:
        text = input("Enter text: ").strip()

        if text.lower() in ("quit", "exit", "q"):
            print("Bye.")
            break

        if not text:
            print("(empty input)\n")
            continue

        tokens = tokenizer(text)
        ids = token_to_id(tokens, vocab)

        if not ids:
            print("(no valid tokens)\n")
            continue

        X = E[ids]                 
        vector = transformer(X)   

        print(f"  Tokens : {tokens}")
        print(f"  IDs    : {ids}")
        print(f"  Vector : shape {vector.shape}")
        print(f"  Values : {np.round(vector[:8], 4)} ...\n")

if __name__ == "__main__":
    main()