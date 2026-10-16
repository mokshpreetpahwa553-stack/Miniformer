MiniFormer

A transformer encoder written from scratch in NumPy.

I built this to understand what happens inside a transformerr. Every part is hand written: softmax, scaled dot product attention, multi head attention, positional encoding, layer normalisation, feed forward layers and residual connections.

Important note on training

All weight matrices are randomly initialised and are not trained. This repo shows the architecture of a transformer encoder and how the pieces fit together. The encoder runs end to end and produces fixed size vectors, but those vectors do not carry learned meaning. Training on a downstream task is future work.


maths_attention.py contains softmax, positional encoding and multi head self attention.

transformer_layer.py contains the feed forward network, layer norm, residual connections, the six layer encoder stack and mean pooling.

transformer.py contains the tokeniser, vocabulary builder, document chunking and end to end encoding.

main.py is a small command line tool. Type a sentence, get a 64 dimensional vector.

-How to run:

git clone https://github.com/mokshpreetpahwa553-stack/MiniFormer.git
cd MiniFormer
pip install -r requirements.txt
python main.py

Type any text when prompted. Type quit to exit.

-Architecture:

Text goes through tokenisation, then vocabulary lookup to token IDs, then embedding lookup of size 64. Positional encoding is added. The result passes through six encoder blocks. Each block does multi head self attention, adds a residual, applies layer norm, runs a feed forward network, adds another residual and applies layer norm again. After six blocks, the output is mean pooled over the token axis to give one vector of size 64.

Every chunk of text becomes one vector of size 64 regardless of length. This is what makes the output usable by vector search libraries such as FAISS.

-Design choices:

NumPy only, because using PyTorch would hide the operations I wanted to learn.

Mean pooling, because vector search needs a fixed dimension per entry.

Four heads and d_model equals 64, small enough to stay readable and large enough to show attention works.

Six layers, to match the original paper.

Post layer norm instead of pre layer norm, also to match the original paper.

-Limitations:

Weights are untrained.

Vocabulary is built from a small Wikipedia corpus and is not general purpose.

No padding mask and no causal mask. This is an encoder only.

-References:

Vaswani et al, Attention Is All You Need, NeurIPS 2017.

Author

Mokshpreet Pahwa
Stage 2 BSc Computer Science, University College Dublin

