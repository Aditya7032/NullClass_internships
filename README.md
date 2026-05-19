# Text-to-Image Generation with GANs and Transformers

This is a deep learning project I built to understand how text-to-image generation actually works under the hood — not just calling an API, but building the pieces from scratch and connecting them together. It goes from basic tokenization all the way up to running Stable Diffusion inference with LoRA fine-tuning.

The progression across the notebooks is intentional. Each one builds on the last, so if you're reading through them in order, you'll see why certain design choices were made and what problems they were solving.

---

## What's in this repo

```
.
├── task1.ipynb                  # WGAN-GP baseline with LSTM text encoder on Flowers102
├── task2.ipynb                      # Same pipeline, now with self-attention and cross-attention
├── task3.ipynb                      # Stable Diffusion v1.5 + LoRA on MNIST captions
├── task4.ipynb                  # Full pipeline: WGAN-GP + SD inference on Oxford Flowers
├── task5_text_embeddings.py           # BERT-style tokenizer and Transformer encoder (standalone)
├── task6_cgan_shape_generation.py     # CGAN trained to generate circles, squares, triangles, stars
└── README.md
```

---

## The idea behind the project

Most tutorials on GANs and diffusion models treat text conditioning as a black box — you pass in a caption and get an image. I wanted to actually understand what's happening at each step: how text gets turned into a vector that a GAN can use, how the generator learns to condition on that vector, and what breaks when you scale things up.

So the project is structured as a learning arc:

- Start by tokenizing text descriptions and encoding them into embeddings using a mini Transformer I built from scratch
- Train a simple CGAN on procedurally drawn shapes (circle, square, triangle, star) to see conditional generation working at the most basic level
- Move up to a real WGAN-GP on Flowers102, where the text conditioning actually has to carry meaningful information about 102 different flower classes
- Add attention mechanisms (both self-attention within the image features, and cross-attention between image features and the text embedding) to see if quality improves
- Finally, swap out the custom GAN for Stable Diffusion and fine-tune it with LoRA to keep GPU memory requirements reasonable

---

## Notebooks

### Assignment 1 — WGAN-GP text-to-image on Flowers102

This is the starting point. The text encoder is an LSTM that takes a token sequence (repeated class label indices, one per timestep) and produces a 128-dimensional embedding. That embedding gets concatenated with a noise vector and fed into a convolutional generator that upsamples from a 4×4 spatial map up to a 64×64 RGB image.

The critic (discriminator) is also conditioned on the text — the image features are flattened and concatenated with the text embedding before the final linear layer. Training uses the WGAN-GP objective rather than vanilla BCE, because BCE tends to saturate early and produce unstable gradients. The gradient penalty term (lambda=10) keeps the critic's Lipschitz constraint satisfied without weight clipping.

One thing worth noting: the optimizer uses `betas=(0.0, 0.9)` rather than the default `(0.9, 0.999)`. This is a recommendation that comes with WGAN — disabling momentum in the first Adam moment helps stabilize the adversarial training loop.

```
Noise dim:          100
Text embed dim:     128
Image size:         64×64
Batch size:         64
Epochs:             40
Learning rate:      1e-4
Lambda GP:          10
Critic iterations:  5 per generator step
```

---

### Class 2 — Adding attention to the generator

The problem with the first notebook is that the generator has no way to model long-range spatial relationships in the image — each pixel is generated somewhat independently of distant pixels. Self-attention fixes this by letting every spatial location attend to every other location.

Cross-attention handles a related issue: the text embedding is only injected once at the input. It doesn't actively guide the image features at deeper layers of the network. The cross-attention module lets the image feature maps query the text embedding at multiple points during upsampling, which keeps the generation semantically grounded throughout.

The self-attention implementation uses 1/8 of the channel dimension for queries and keys to keep memory usage down — this is the same trick used in the original SAGAN paper. The cross-attention module is lighter: it treats the text embedding as a single key-value pair and uses dot-product attention to produce a spatially uniform correction term added back to the feature map.

```
ConvTranspose → BatchNorm → ReLU
    → SelfAttention
    → CrossAttention(image features, text embedding)
    → ConvTranspose → BatchNorm → ReLU
    → ConvTranspose → Tanh
```

---

### Class 3 — Stable Diffusion with LoRA

At this point the custom GAN starts hitting its ceiling — 102 flower classes with limited data per class is genuinely hard for a GAN to handle well. So this notebook switches to Stable Diffusion v1.5 as the backbone.

The dataset is MNIST with text captions like `"a handwritten digit seven"`. Each image is resized to 512×512 and normalized to the range SD expects. The VAE encodes images into the latent space, noise is added according to the diffusion schedule, and the UNet is trained to predict the noise given the noisy latent and the text embedding from the CLIP text encoder.

Training the full UNet is expensive (~22 GB VRAM), so I used LoRA to inject low-rank weight updates into just the attention projection layers (`to_q`, `to_k`, `to_v`). With rank r=8, this means fine-tuning roughly 0.1% of the UNet's parameters while leaving the rest frozen. Two epochs is enough to see the model starting to adapt.

```
LoRA rank:          8
Alpha:              16
Target modules:     to_q, to_k, to_v
Dropout:            0.1
Optimizer:          AdamW, lr=1e-4
```

---

### Assignment 4 — Putting it all together on Oxford Flowers 102

This is the most complete notebook. Part 1 trains a WGAN-GP on the actual Oxford Flowers 102 dataset with a proper word-level vocabulary built from the flower category names. The resolution goes up to 96×96, the noise and text embedding dims are larger (256 and 512 respectively), and the vocabulary covers 4096 tokens.

Part 2 loads Stable Diffusion v1.5 with CPU offload (to keep peak VRAM around 3-4 GB on a T4) and uses it for inference with a `generate_flower()` utility function. You can pass any flower name and pick from four styles:

```python
generate_flower("sunflower",     style="photorealistic", seed=42)
generate_flower("pink rose",     style="watercolor",     seed=7)
generate_flower("purple orchid", style="oil painting",   seed=123)
generate_flower("daisy",         style="sketch",         seed=99)
```

The DPMSolver scheduler gets acceptable quality in 25 steps, which is roughly equivalent to 50 DDIM steps but twice as fast.

---

## Task 1 — Text preprocessing and embeddings

`task1_text_embeddings.py` / `Task1_Text_Embeddings.ipynb`

Before plugging text into any model, I wanted to understand what tokenization and encoding actually look like step by step. This script builds a BERT-style pipeline from scratch — no pretrained weights, no HuggingFace downloads.

The vocabulary covers about 60 tokens: special tokens (`[CLS]`, `[SEP]`, `[PAD]`, `[UNK]`, `[MASK]`) plus words from shape descriptions like colors, geometry terms, and spatial prepositions. The tokenizer uses a simple regex to split words, wraps the sequence with `[CLS]` and `[SEP]`, and pads to a fixed length of 16 tokens.

The encoder is a 2-layer Transformer with 4 attention heads and sinusoidal positional encodings. The `[CLS]` token's hidden state at the final layer is used as the sentence embedding — same convention as BERT.

Running it on 8 shape descriptions like `"A red circle on a white background"` produces 128-dimensional vectors for each. The notebook visualizes these as a heatmap, plots them in 2D using PCA, shows the cosine similarity between all pairs, and breaks down the L2 norm contribution per token.

---

## Task 2 — Conditional GAN for shapes

`task2_cgan_shape_generation.py` / `Task2_CGAN_Shape_Generation.ipynb`

This is the simplest possible version of a conditional GAN — trained on procedurally drawn 32×32 shapes with four classes: circle, square, triangle, star. 200 samples per class, with a small amount of Gaussian noise added so the generator has to learn something real rather than just memorizing pixel-perfect templates.

Both the generator and discriminator receive a label embedding that gets concatenated to their inputs. The generator takes a 64-dimensional noise vector plus the label embedding and passes it through three fully connected layers before reshaping to an image. The discriminator flattens the image and concatenates the label embedding before its own FC stack.

After 150 epochs the generator produces clearly distinct shapes for each label. The notebook saves snapshots at six checkpoints (epochs 1, 10, 30, 60, 100, 150) so you can watch the shapes get sharper over training.

```
Epochs:              150
Batch size:          64
Learning rate:       2e-4
Optimizer:           Adam (β₁=0.5, β₂=0.999)
Loss:                Binary Cross-Entropy
Samples per class:   200
```

---

## How it all fits together

At a high level, every notebook in this repo is doing some version of the same thing:

1. A text input gets tokenized and encoded into a fixed-size vector
2. That vector gets passed to a generator alongside a noise sample
3. The generator produces an image
4. A discriminator (or diffusion denoising network) judges the quality and pushes the generator to improve

The differences between notebooks are mostly about where the text conditioning happens, how expressive the text encoder is, and what loss function keeps training stable. Going from BCE to WGAN-GP to diffusion is essentially a story of finding better ways to define "the generator improved."

---

## Running it yourself

```bash
git clone https://github.com/<your-username>/<repo-name>.git
cd <repo-name>

python -m venv venv
source venv/bin/activate

pip install torch torchvision numpy matplotlib scikit-learn
pip install transformers diffusers accelerate peft safetensors pillow jupyter
```

Task 1 and Task 2 run on CPU and finish in a few minutes. The GAN notebooks will be slow on CPU — a GPU makes a real difference. The Stable Diffusion parts need at least 4 GB VRAM.

```bash
# standalone scripts
python task1_text_embeddings.py
python task2_cgan_shape_generation.py

# or open the notebooks
jupyter notebook
```

---

## Libraries used

PyTorch and torchvision handle all the model code and standard datasets. The Stable Diffusion notebooks use HuggingFace Diffusers and PEFT for the LoRA integration. Matplotlib covers all the visualizations, scikit-learn provides PCA for the embedding plots, and NumPy handles the procedural shape generation in Task 2.

---

## License

This project is for learning purposes. The Stable Diffusion weights (`runwayml/stable-diffusion-v1-5`) are covered by the [CreativeML Open RAIL-M license](https://huggingface.co/spaces/CompVis/stable-diffusion-license). Oxford Flowers 102 is a dataset from the Visual Geometry Group at the University of Oxford.
