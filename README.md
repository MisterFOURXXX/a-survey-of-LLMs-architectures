# A Survey of LLM Architectures

[![Python](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1%2B-orange)](https://pytorch.org/)
[![Transformers](https://img.shields.io/badge/transformers-4.40%2B-yellow)](https://huggingface.co/docs/transformers)
[![PEFT](https://img.shields.io/badge/peft-0.10%2B-green)](https://huggingface.co/docs/peft)
[![License: MIT](https://img.shields.io/badge/License-MIT-lightgrey)](LICENSE)

---

## Table of Contents

- [Abstract](#abstract)
- [Introduction](#introduction)
- [Objective](#objective)
- [Foundations](#foundations)
  - [Large Language Model (LLM)](#large-language-model-llm)
  - [LLMs Components](#llms-components)
  - [LLMs Model Selection](#llms-model-selection)
  - [LLMs Suffix Syntax](#llms-suffix-syntax)
  - [Fine Tuning LLMs](#fine-tuning-llms)
  - [HuggingFace API Parameters](#huggingface-api-parameters)
  - [Model Family Taxonomy](#model-family-taxonomy)
- [Models Surveyed](#models-surveyed)
- [Repository Structure](#repository-structure)
- [Getting Start](#getting-start)
  - [Kaggle API Credentials](#kaggle-api-credentials)
  - [Environment Installation](#environment-installation)
  - [Data Preparation](#data-preparation)
  - [Running Experiments](#running-experiments)
  - [Configuration Reference](#configuration-reference)
- [Results and Metrics](#results-and-metrics)
- [Citation and References](#citation-and-references)
- [License](#license)

---

## Abstract

This repository is a **reproducible research workbench** for studying, fine-tuning, and
evaluating modern Large Language Models (LLMs) across five architectural families:

1. **GPT-style dense decoders** (Pythia, GPT-2, Llama 3.1, Qwen 3)
2. **Mixture-of-Experts (MoE) transformers** (Phi-3.5-MoE, DeepSeek-V2-Lite, Qwen1.5-MoE)
3. **Hybrid SSM–Transformer models** (AI21 Jamba 2)
4. **Large Reasoning Models (LRMs)** (Qwen3-Thinking, DeepSeek-R1-Distill, Phi-4-mini-reasoning)
5. **Small Language Models (SLMs)** (Phi-3.5-mini, Gemma-3-1B, SmolLM3-3B)

Every experiment is driven by a YAML configuration, orchestrated through a single modular
Python package, and evaluated with both intrinsic (perplexity) and extrinsic (ROUGE,
BLEU, generation-strategy comparison) metrics. The codebase is intentionally small so
that each architectural idea can be read end-to-end.

## Introduction

Welcome to this comprehensive tutorial on Large Language Models (LLMs). This notebook serves as a practical guide to understanding, fine-tuning, optimizing, and evaluating state-of-the-art LLMs. The content explores the architectural foundations of modern language models, dives into various fine-tuning techniques, and implements hands-on experiments with different model families.

Large Language Models represent a paradigm shift in artificial intelligence, enabling machines to understand, generate, and manipulate human language with unprecedented sophistication. These models have evolved from simple pattern matchers to complex reasoning systems capable of handling diverse tasks across multiple domains.

This tutorial covers:
- Dive deep into specific LLMs model architecture and components of modern LLMs.
- Dive deep into different model families and specialized applications.
- Fundamental implementation of fine-tuning techniques on various pre-trained models.
- Fundamental optimization strategies.
- Evaluation of model performance using multiple metrics.

## Objective

**Primary Objectives:**

- Architectural Understanding: Comprehension of core components in Transformer-based LLMs, including attention mechanisms, positional encodings, and normalization layers.
- Model Selection: Guidance on choosing appropriate models for specific NLP tasks based on architectural characteristics.
- Fine-tuning Proficiency: Mastery of multiple fine-tuning techniques, ranging from full parameter updates to parameter-efficient methods like LoRA and QLoRA.
- Optimization Skills: Implementation and comparison of various optimization strategies for efficient training.
- Evaluation Competence: Application of comprehensive evaluation metrics to assess model performance.

**Technical Objectives:**

- Implementation of data preprocessing pipelines for LLM training.
- Configuration and execution of fine-tuning for different model architectures.
- Optimization of training for memory efficiency and computational performance.
- Comparison of generation strategies and their impact on output quality.
- Analysis of model performance across multiple evaluation dimensions.

---

## Foundations

### Large Language Model (LLM)

A Large Language Model (LLM) is a type of Artificial Intelligence trained to understand, generate, and manipulate human language. These models are "large" in two senses: they are trained on massive datasets (nearly the entire public internet) and contain billions of internal mathematical connections, known as parameters.

**How does LLMs worked?**

At their core, Large Language Models (LLMs) function as probabilistic inference engines based on the Transformer architecture. Unlike rule-based systems, an LLM does not possess conceptual understanding; rather, it performs autoregressive next-token prediction. By analyzing the statistical distributions within vast datasets, the model estimates the conditional probability of a subsequent unit of text given a specific input sequence.

The transition from raw human language to machine-generated output involves several distinct computational stages:

- Tokenization: This is the initial preprocessing phase where discrete text is decomposed into tokens—the fundamental atomic units of the model. These units may represent whole words, sub-words, or individual characters, allowing the model to handle a diverse vocabulary and out-of-vocabulary terms efficiently.
- Vector Embedding: Because neural networks cannot process raw text, tokens are mapped into a high-dimensional continuous vector space. These embeddings capture semantic and syntactic relationships, ensuring that tokens with similar contextual usage are mathematically proximal to one another.
- Self-Attention Mechanism: This is the defining feature of the Transformer architecture. The attention mechanism allows the model to calculate the relevance of every token in a sequence relative to every other token. By utilizing Query (Q), Key (K), and Value (V) matrices, the model identifies long-range dependencies, enabling it to maintain context across extensive passages.
- Probabilistic Prediction: In the final layer, the model applies a Softmax function across its entire vocabulary. This produces a probability distribution where the sum of all possible next tokens equals 1. The model then selects the most statistically probable token to append to the sequence, repeating this cycle until a termination condition is met.

**Modern LLMs**

Modern AI agents aren’t powered by "one model to rule them all." Instead, developers mix and wire together different model types—each optimized for a specific job. This modular approach allows for smarter resource allocation, lower latency, and more reliable reasoning across complex workflows.

**The 8 Categories of LLMs**

- GPT (General-Purpose Generative Pretrained Transformer) : This is the default "text-in to text-out" brain used for general reasoning, drafting, and synthesis. While it offers the strongest generalization and is easy to steer, it can often be overkill and expensive for simple, repetitive tasks. It serves as the primary engine for most open-ended conversational applications.
- MoE (Mixture of Experts) : Designed for high throughput and scale, this architecture uses a "router" to activate only a few specific "expert" sub-networks for each token processed. This allows the model to have a massive effective parameter count without requiring the full compute cost for every step. It is the go-to "big-iron" upgrade when you need scale without crushing latency.
- LRM (Large Reasoning Model) : Focused on multi-step reasoning, tool use, and Retrieval-Augmented Generation (RAG). These models utilize chain-of-thought processing and self-consistency checks to provide search-grounded, verifiable answers with citations. They are essential for complex technical queries where reliability is more important than speed.
- VLM (Vision-Language Model) : These models interpret pixels and text simultaneously to understand screenshots, diagrams, and physical documents. They turn visual data into structured insights, making them ideal for processing invoices, navigating user interfaces, or analyzing product photos.
- SLM (Small Language Model) : Optimized for on-device or edge computing, SLMs are fast, cheap, and private. They are perfect for fast control loops and local data governance where a 70B parameter model would be too slow or resource-intensive. They require tighter prompts but excel at narrow, well-defined tasks.
- LAM (Large Action Model) : A LAM is specialized in turning language into concrete actions via tools and APIs. Whether it is booking a flight, creating a JIRA ticket, or querying a database, these models focus on executing real-world work through deterministic schemas and strict input validation.
- HLM (Hierarchical Language Model) : This operates as a "planner–worker" stack where a high-level top model decomposes complex tasks into a list of sub-tasks. Specialist models then execute those individual steps. While it introduces orchestration complexity, it provides high transparency, modularity, and the ability to parallelize work.
- LCM (Large Concept Model) : Instead of generating text immediately, an LCM maps inputs to interpretable concepts like specific skills, entities, or risk categories. This makes the AI's decision-making process auditable and controllable, allowing it to be easily grounded in Knowledge Graphs or strictly enforced safety policies.

### LLMs Components

#### Tokenization

In 2026, tokenization is no longer just "breaking words into pieces"; it is a highly engineered compression layer. As vocabularies expand to 128k or 256k entries, the efficiency of how text is mapped to integers directly determines the model's "effective" context length and reasoning speed.

**Byte-Pair Encoding (BPE) and Tiktoken**

BPE remains the industry standard, notably through the Tiktoken implementation. It iteratively merges the most frequent adjacent pairs of bytes into a single new token.
- Large Vocabularies: By using massive vocabularies (e.g., GPT-4o and Llama 3/4), models can represent entire technical terms (e.g., "photosynthesis") or common code snippets as a single integer. This reduces the total sequence length, allowing the model to "see" more text in its fixed context window.

**WordPiece and SentencePiece**

While BPE is common in decoder-only models, WordPiece (used by BERT/Gemma) and SentencePiece (used by Llama/T5) offer different advantages.
- SentencePiece is unique because it treats the input as a raw stream of characters—including spaces—which it treats as a special symbol (_). This removes the need for a language-specific "pre-tokenizer," making it the architecture of choice for truly global, multilingual models that handle languages like Japanese or Chinese where word boundaries aren't marked by spaces.

**Unigram Language Modeling** 

Unlike BPE, which starts with characters and merges them, Unigram starts with a massive vocabulary and iteratively prunes the least useful tokens based on a probabilistic model.
- The Benefit: It provides multiple ways to tokenize a single sentence (subword regularization), which can be used during training to make the model more robust to spelling errors or different writing styles.

**Byte-level and Character-level Fallbacks**

To eliminate "out-of-vocabulary" errors, modern tokenizers use UTF-8 Byte Fallback. If a model encounters a rare emoji or an obscure mathematical symbol it doesn't recognize, it breaks that character down into its raw UTF-8 bytes. This ensures the model can process any possible input string without crashing or returning an <UNK> (unknown) token.

**Visual and Multimodal Tokenization**
    
The 2026 generation of "Native Multimodal" models uses Visual Quantization (such as VQ-VAE or ViT patches).
- Discrete Vision Tokens: Images are sliced into small patches, and each patch is mapped to a "visual vocabulary" entry. To the model's internal Transformer, an image is simply a sequence of tokens, no different from a paragraph of text, enabling seamless cross-modal reasoning within a single unified latent space.

| Algorithm | Used by | Key Idea |
|-----------|---------|----------|
| **Byte-Pair Encoding (BPE)** | GPT-2, GPT-4 (Tiktoken), Llama 3, Qwen 3 | Iteratively merge the most frequent adjacent byte pairs. |
| **WordPiece** | BERT, Gemma | Similar to BPE but merges by likelihood rather than frequency. |
| **SentencePiece / Unigram** | Llama 2, T5, Gemma-3 | Operates on a raw character stream (`_` for space); supports subword regularization. |
| **Tiktoken (large-vocab BPE)** | GPT-4o (200k), Llama 3 (128k), Qwen 3 (152k), Gemma-3 (256k) | Whole technical terms → a single token; dramatically reduces sequence length. |
| **UTF-8 Byte Fallback** | All modern tokenizers | Guarantees no out-of-vocabulary tokens by falling back to raw bytes for rare codepoints. |

#### Attention Mechanisms

The standard Multi-Head Attention (MHA) was the primary bottleneck of early models due to its quadratic scaling. Modern architectures utilize a suite of specialized mechanisms to handle massive contexts and noise reduction:

- Grouped-Query Attention (GQA): Used in Llama 3/4 and Mistral. It functions as a middle ground between MHA and Multi-Query Attention (MQA). By grouping multiple query heads to share a single Key-Value (KV) head, it drastically reduces the memory footprint of the KV Cache. This allows for significantly higher throughput and longer conversations without the performance degradation seen in MQA.
- Differential Attention (Diff-Transformer): A breakthrough from Microsoft Research that tackles "attention noise." Instead of a single softmax map, it calculates the difference between two separate attention maps. This subtraction cancels out irrelevant background noise, allowing the model to focus purely on the "signal," which significantly reduces hallucinations in long documents.
- Ring Attention: Designed for "near-infinite" context, Ring Attention distributes the attention calculation across a ring of multiple GPUs. By overlapping the communication of KV blocks with the computation of the next block, it allows models to process sequences of 100 million tokens or more, limited only by the number of devices in the cluster rather than individual GPU memory.
- Dilated Attention (LongNet): Inspired by dilated convolutions in vision, this mechanism expands the model’s "attentive field" exponentially as the distance between tokens increases. It provides a logarithm dependency between tokens, enabling a $1$ billion token context window while maintaining linear computational complexity.
- Flash Attention-3: A hardware-aware algorithm that optimizes how the GPU’s SRAM and HBM interact. By using asynchronous execution and "warpshaping" strategies, it eliminates memory-bound bottlenecks, making the $O(N^2)$ calculation so fast that context windows of 128k to 1M tokens become standard even on consumer-grade hardware.
- Sliding Window Attention (SWA): Rather than attending to every token in a document, each layer only "sees" a fixed-size window of nearby tokens. Because these windows are stacked across layers, the model develops a "receptive field" that covers the entire sequence, reducing complexity from $O(N^2)$ to $O(N)$.
- Multi-Head Latent Attention (MLA): Popularized by the DeepSeek-V3 series, MLA uses low-rank joint compression for the KV cache. It compresses the Key and Value vectors into a latent space, allowing the model to maintain the representational power of many heads while requiring only a fraction of the memory traditionally needed to store the conversation history.

| Variant | Where used | Trade-off |
|---------|------------|-----------|
| **Grouped-Query Attention (GQA)** | Llama 3.1, Qwen 3, Falcon 3, Phi-3.5, SmolLM3, Gemma 3 | Multiple Query heads share one KV head → ~8× smaller KV cache, near-identical quality. |
| **Multi-Head Latent Attention (MLA)** | DeepSeek-V2-Lite, DeepSeek-R1 | Compress K and V into a low-rank latent vector; up-project at inference. |
| **Sliding Window Attention (SWA)** | Mistral, Ministral 3, Falcon | Each token attends to a fixed window of the past; receptive field grows with depth. |
| **Flash Attention-3** | All 8B+ training runs | IO-aware kernel; makes attention ~2–4× faster on Ampere/Hopper. |
| **Ring / Dilated Attention** | Reference-only | Distributes attention across GPUs or expands the field exponentially for 1M+ contexts. |

#### Positional Encodings

Positional encodings are the "GPS" of a Large Language Model. Because the Transformer architecture processes all words in a sequence simultaneously (parallelization), it has no inherent sense of order. Without these encodings, a model would see "The dog bit the man" and "The man bit the dog" as identical sets of words. Detailed breakdown of the primary positional encoding strategies used in modern LLMs:

**1. Absolute Positional Encodings**

These methods assign a unique, fixed vector to every specific index (e.g., position 1, position 2).
- Sinusoidal (Fixed) Encodings: Used in the original 2017 Transformer. It uses sine and cosine functions of different frequencies to create a unique signature for each position.
    - Pros: Requires no extra parameters; can theoretically handle sequences longer than those seen during training.
    - Cons: Struggles to capture complex relative relationships between tokens.
- Learned Encodings: Used in models like BERT and GPT-2. Instead of a formula, the model has a dedicated "lookup table" of embeddings for each position (up to a maximum limit, like 512 or 2048) that it optimizes during training.
    - Pros: Can learn the most "useful" representation for a specific dataset.
    - Cons: Hard-coded limit; the model cannot process a single token beyond the maximum position it was trained on.

**2. Relative Positional Encodings**

Rather than focusing on where a token is in the sentence, these focus on how far tokens are from each other.
- ALiBi (Attention with Linear Biases): Popularized by the MPT and Bloom models. It doesn't add vectors to the embeddings. Instead, it penalizes the attention score between two tokens based on their distance—the further apart they are, the lower the bias.
    - Pros: Incredible extrapolation; a model trained on 2k tokens can often work on 8k or more without retraining.
- T5 Relative Bias: Used in Google's T5 model. It learns a specific scalar bias for different "buckets" of distances (e.g., "1 token away," "2–4 tokens away," etc.).

**3. Rotary Positional Embeddings (RoPE)**

The current industry standard for state-of-the-art models like Llama 3, PaLM, and Mistral.
- Mechanism: RoPE treats token representations as pairs of coordinates in a high-dimensional space and "rotates" them. The amount of rotation depends on the token's position. When the model calculates attention, the dot product between two tokens naturally captures their relative distance through the angle between them.
- Scaling (Long Context): RoPE is highly compatible with "Linear Scaling" and YaRN (Yet another RoPE extension), allowing developers to "stretch" the rotation angles to expand context windows from 4k to 128k or even 1M tokens with minimal fine-tuning.

**4. Contextual & Advanced Encodings (The Frontier)**

Newer methods aim to make "position" more than just a number.
- CoPE (Contextual Position Encoding): A 2024 advancement from Meta. Instead of counting tokens (1, 2, 3...), CoPE allows the model to "count" based on context. For example, it can learn to attend to the "2nd previous sentence" or the "3rd previous code function," regardless of how many tokens are in between.
- NoPE: Some research suggests that with specific training (like in the C4 dataset), models can actually learn implicit positional logic without any explicit encoding at all, though this is not yet the standard for large-scale production models.

| Family | Example | Note |
|--------|---------|------|
| **Sinusoidal (fixed)** | Original 2017 Transformer | No parameters; poor relative reasoning. |
| **Learned absolute** | GPT-2 | Lookup table; hard-coded max length. |
| **ALiBi** | MPT, Bloom | Distance-based penalty; excellent extrapolation. |
| **RoPE (Rotary)** | Llama 3, Qwen 3, Falcon 3, Phi-3.5, Gemma 3 | Rotates Q/K vectors by position-dependent angles; current industry standard. |
| **RoPE + YaRN / LongRoPE** | Llama 3.1 (128k), Phi-3.5 (128k) | "Stretch" rotation angles to expand context with minimal fine-tuning. |
| **CoPE (contextual)** | Research frontier | Counts by *semantic* units (e.g. "2nd previous sentence"). |

#### Architecture

In early 2026, the landscape of Large Language Model (LLM) design has matured into a diverse ecosystem of specialized primitives. Developers no longer rely on raw scaling alone; instead, they utilize a sophisticated array of architectural families to navigate the trade-offs between "reasoning density," memory bandwidth, and the quadratic costs of long-sequence processing.

**Dense Transformers (The Baseline)**

The classic decoder-only Transformer remains the foundation for mid-sized models (7B–70B parameters) where maximum knowledge density is required. In this setup, every input token activates every parameter in the network, ensuring predictable scaling and high-quality reasoning.
- The Components: Modern dense stacks use RMSNorm or LayerNorm variants for numerical stability during training and SwiGLU/GeGLU activations in the feed-forward blocks to capture non-linear relationships more effectively.
- Positioning: Instead of fixed sine waves, these models use Rotary Positional Embeddings (RoPE) or ALiBi, allowing the model to understand the relative distance between words, which is crucial for maintaining coherence in long documents.
- Examples: Llama 3.1/4 (Meta), Gemma 3 (Google), and Qwen 2.5 (Alibaba).

**Mixture-of-Experts (MoE)**

MoE has become the dominant paradigm for frontier-class models reaching into the trillions of parameters. Instead of a single massive feed-forward layer, MoE uses a "sparse" architecture containing many parallel sub-networks called "experts."
- How it Works: A lightweight Gating Network (Router) evaluates each token and directs it to only a few specific experts (usually 1 to 8). This allows a model to have a massive "brain" (total parameters) while only spending the compute power of a much smaller model (active parameters) for each word generated.
- Trade-offs: While MoE offers incredible throughput, it requires sophisticated load-balancing to prevent "router collapse" (where one expert gets overworked while others sit idle) and carries a larger memory footprint, as the entire expert suite must be available in VRAM.
- Examples: DeepSeek-V3, Mixtral 8x22B, and Grok-3.

**State-Space Models (SSMs) and Mamba**

Introduced as a challenge to the "Attention" bottleneck, Mamba-family models replace the standard self-attention mechanism with Selective Scanning. This allows the model to process sequences with linear-time complexity ($O(N)$) rather than the squared complexity ($O(N^2)$) of Transformers.
- The "State" Advantage: These models maintain a compressed internal state that is updated sequentially, making them incredibly efficient for ultra-long contexts (100k to 1M+ tokens).
- Evolution: Early SSMs struggled with precise data retrieval, but by 2026, improved selective mechanisms have allowed them to rival Transformers in reasoning while using a fraction of the memory for long-form generation.
- Examples: Jamba 1.5 (AI21 Labs) and Falcon Mamba.

**Diffusion Transformers (DiT) for Text and Multimodality**

Originally designed for image generation, the Diffusion Transformer (DiT) architecture is increasingly used for LLMs that must handle multimodal inputs (video, audio, text) within a single unified latent space.
- The Architecture: These models treat text generation as a denoising process. By using a Transformer backbone to predict noise levels, they offer smoother transitions in creative writing and better integration with visual data.
- Scaling: DiT scales exceptionally well with compute, making it a favorite for models that power generative video and world simulators.
- Examples: Sora-v2 (OpenAI) and Llama-Gen (Meta).

**Retentive Networks (RetNet)**

RetNet is a strong competitor to both Transformers and SSMs, marketed as "The Successor to Transformers." It utilizes a Multi-Scale Retention mechanism that can operate in three modes: parallel (for fast training), recurrent (for efficient inference), and chunkwise (for long-form processing).
- The Mechanism: It avoids the $O(N^2)$ cost by using a decaying retention score that naturally forgets older, less relevant information while keeping a sharp focus on recent context.
- Inference Speed: RetNet offers significantly higher throughput than Transformers because it doesn't require a traditional, massive KV cache to function.
- Examples: Research models from Microsoft and specialized "Edge LLMs."

**Hybrid Architectures**

The most promising frontier in 2026 is the hybrid approach, which interleaves different layer types to neutralize the weaknesses of individual architectures. By combining Attention, MoE, and SSM layers, developers create models that are "jacks of all trades."
- The Synergy: Attention excels at global reasoning and precise retrieval; MoE provides cheap parameter scaling for vast knowledge; and SSMs/Mamba layers provide memory efficiency for massive context windows.
- Implementation: A model might use one Attention layer for every three Mamba layers, ensuring the model can "look back" precisely while maintaining high-speed streaming capabilities.
- Examples: DeepSeek-V3 (MoE + Dense), Jamba (Transformer + Mamba + MoE), and Qwen 3 experimental hybrids.

#### Normalization and Activation Functions

**Normalization**

Normalization acts as the model's "internal thermostat," ensuring that signal values do not explode or vanish as they pass through hundreds of layers.
- RMSNorm (Root Mean Square Layer Normalization): Now the universal standard for models like Llama 3, GQA-based hybrids, and DeepSeek. Unlike traditional LayerNorm, it skips the "mean-centering" step and only scales the input by the root mean square of its elements.
    - Architecture Benefit: It provides a 10%–40% speedup in the normalization phase because it avoids the overhead of calculating the mean. It assumes that the "offset" (mean) of the data is less important than its "scale" (variance) for stabilizing gradients.
- DeepNorm: Developed to solve the "Deep Transformer" problem where models with 100+ layers become unstable. It modifies the initialization and residual connections, scaling the identity branch differently than the sub-layer branch.
    - Architecture Benefit: It allows models to scale to extreme depths (e.g., 200–400 layers) without requiring the complex hyperparameter tuning usually needed to prevent training collapse.
- LayerNorm (Standard): Still found in some older GPT-style architectures. It normalizes across the feature dimension and includes a "bias" term. In 2026, it is largely considered legacy for frontier LLMs due to higher computational overhead compared to RMSNorm.
- Query-Key Normalization (QK Norm): A specialized technique where normalization is applied specifically to the Query and Key vectors inside the attention head.
    - Architecture Benefit: It prevents "attention sinks" and numerical instability in very high-dimensional models, ensuring that the dot-product results don't exceed the precision limits of FP16 or BF16 formats.
    
**Activation Functions**

Activation functions are the "gates" that decide which information is important enough to pass to the next layer. Modern functions are "gated," meaning they use one part of the input to decide how much of the other part to let through.
- SwiGLU (Swish-Gated Linear Unit): The 2026 gold standard for feed-forward networks (FFN). It is a hybrid of the Swish function and the Gated Linear Unit.
    - Architecture Benefit: Instead of a simple "on/off" switch like ReLU, SwiGLU uses two separate linear layers that are multiplied together. This provides a smoother gradient and more "expressive power," allowing the model to learn subtle nuances in logic and coding.
- GeGLU (Gaussian Error Gated Linear Unit): A variant of the GELU activation used in many Google-originated models like PaLM and Gemini.
    - Architecture Benefit: It approximates the Gaussian cumulative distribution function. Like SwiGLU, it is a gated unit, but it utilizes the Gaussian error function to "soften" the activation, which has been shown to improve performance in reasoning-heavy tasks.
- GELU (Gaussian Error Linear Unit): The successor to ReLU used in early GPT-3/4 models. It weights inputs by their percentile rather than just zeroing out negative values.
    - Architecture Benefit: It allows a small amount of "negative signal" to pass through, which helps prevent "dead neurons" (where parts of the model stop learning entirely during training).
- ReLU (Rectified Linear Unit): The classic $f(x) = \max(0, x)$ function. In 2026, it is almost exclusively used in tiny "edge" models or older research benchmarks because it lacks the sophistication required for complex language understanding at scale.

### LLMs Model Selection

**NLP Tasks and LLMs**

**1. Text Summarization (Long-Context Specialists)**

Summarization requires a model that can ingest massive amounts of data without suffering from "middle-of-the-document" forgetfulness, a phenomenon known as the "lost in the middle" problem.
- Jamba 1.5/2.0 (AI21 Labs): A hybrid architecture that interleaves Transformer layers with Mamba (SSM) layers. This allows for linear-time complexity $O(N)$, meaning it can summarize a 500-page document with the same efficiency a standard model uses for a few pages.
- Claude 4 (Anthropic): Utilizing ALiBi and RoPE positional encodings, Claude is optimized for "high-fidelity retrieval." It is excellent for creative or executive summaries where the "tone" of the original text must be preserved.
- Gemini 1.5 Pro: Its Ring Attention mechanism allows it to process up to 2 million tokens, making it the only choice for summarizing entire video series or massive technical repositories in a single pass.

**2. Language Translation (Multilingual & Sparse Models)**

Effective translation requires a massive "world knowledge" base and a tokenizer that treats all languages equally.
- DeepSeek-V3: A Mixture-of-Experts (MoE) model that excels at high-speed translation. It uses Multi-Head Latent Attention (MLA) to reduce memory costs while maintaining trillion-parameter performance across 100+ languages.
- Qwen 3 (Alibaba): The dominant model for Eastern linguistic nuances. Its architecture is optimized with RMSNorm and a massive multilingual training set, specifically excelling in the transition between high-context languages like Mandarin and low-context languages like English.

**3. Chatbots & Conversational AI (High-Throughput Models)**

Chatbots require low latency and efficient handling of the "KV Cache" to remember long conversations without crashing the server or increasing cost-per-query.
- GPT-4o / GPT-5 (OpenAI): A native multimodal model. Its breakthrough is a Unified Latent Space, allowing it to process text, audio, and vision simultaneously. This makes it the premier choice for human-like, real-time voice and vision-integrated chatbots.
- Llama 4 (Meta): The industry standard for self-hosted chat. Because it utilizes Grouped-Query Attention (GQA), the memory required to store conversation history is slashed, allowing a single GPU to serve dozens of concurrent users.

**4. Information Extraction & Topic Modeling**

These tasks require a model to strictly adhere to logical structures (like JSON or SQL) and identify latent themes in messy, unstructured data.
- Mistral Large 3: Known for its "instruction-following" precision. It uses a refined SwiGLU activation setup that excels at identifying entities (names, dates, prices) and extracting them into structured formats without "hallucinating" extra data.
- BERTopic-LLM (Hybrid): For topic modeling, developers now use a hybrid approach: an encoder model (like RoBERTa-v2) to create embeddings, and a generative model (like Llama 4) to label the discovered clusters with natural language names.

**5. Text Classification & Sentiment Analysis**

Classification is a "discriminative" task that is now handled most efficiently by small-language models (SLMs) or "Distilled" models that minimize latency.
- Phi-4 (Microsoft): A tiny but mighty model (under 4B parameters) trained on "textbook-quality" synthetic data. It is the gold standard for high-volume sentiment analysis, spam detection, and intent classification on local or edge hardware.
- Gemma 3 (Google): Built on the same technical foundations as Gemini, this model is optimized for on-device execution, ensuring that sensitive data used for classification (like internal company emails) never leaves the local machine.

**6. Question Answering (QA) & RAG (Retrieval-Augmented Generation)**

Modern QA models focus on "grounding" their answers in external evidence to prevent hallucinations.
- Perplexity-Sonar (Online LLMs): These are specialized models optimized for searching the live web. They use Cross-Attention mechanisms to compare their internal knowledge against real-time search results, citing sources with high precision.
- Command R+ (Cohere): Specifically designed for RAG workflows. It has a high "Contextual Precision" score, meaning it excels at ignoring irrelevant data in a search result and focusing only on the snippets that answer the user's query.

**7. Natural Language Inference (NLI) & Formal Logic**

NLI involves determining if one sentence logically follows from another (entailment, contradiction, or neutral).
- DeepSeek-R1 / o1-series: These use "Chain of Thought" (CoT) reinforcement learning. They are the best for tasks where the model must "verify" its own logic before providing an answer, making them ideal for legal reasoning or software bug detection.

**8. Text-to-SQL & Code Generation**

These models treat natural language as a prompt to generate rigid, executable syntax.
- StarCoder 3: A model trained almost exclusively on code and technical documentation. It uses Fill-In-the-Middle (FIM) capabilities, allowing it to complete code in the middle of a file by looking at both the code above and below the cursor.

### LLMs Suffix Syntax

Understanding this "suffix syntax" is essential to knowing whether a model will fit on your hardware, how it will behave in a chat, and which architectural breakthroughs it utilizes.

**1. The Llama & Meta Collection**

In the Meta ecosystem, the suffix indicates the scale of the "brain" and its specialized protective layers.

Model Size (1B, 3B, 8B, 11B, 70B, 405B): The number represents parameters in billions ($10^9$).

- 1B – 3B: "Mobile/Edge" versions. Optimized for on-device processing (phones/tablets).
- 8B – 11B: "Small/Workstation" versions. The 11B variant often includes Vision capabilities. These run on high-end consumer GPUs (RTX 4090/5090).
- 70B: "Medium/Pro" version. The balanced choice for enterprise-grade reasoning.
- 405B: "Flagship/Frontier" version. Meta’s most powerful model, requiring massive computing clusters.

Model Type (Base vs. Instruct):

- -Base / No Suffix: The raw "Next-Token Predictor." Trained to complete patterns but lacks social conditioning; it may repeat questions rather than answering them.
- -Instruct: The "Agentic" version. Refined via SFT and RLHF, it is designed to follow commands and behave as a helpful assistant.

Safety & Capability Suffixes:

- -Llama-Guard: A "Classifier" model acting as a digital bouncer to detect toxic, biased, or dangerous content.
- -Prompt-Guard: A tiny 86M parameter model designed to intercept "Prompt Injections" and jailbreak attempts.
- -Vision: Indicates the model is multimodal, capable of processing image inputs alongside text.

**2. The GPT & OpenAI Series**

OpenAI uses suffixes to denote specific optimization paths, ranging from all-purpose models to high-speed thinkers.

Model Flavors:

- -o (Omni): A native multimodal model handling text, audio, and vision in a single unified space.
- -turbo: Versions optimized for significantly lower latency and higher throughput.
- -mini: A smaller, cost-efficient version with high "reasoning density" for its scale.

Reasoning & Snapshots:

- -o1 / -o3: Reasoning models using Chain-of-Thought (CoT). They "think" before generating an answer to verify logic.
- Date-Based Snapshots (e.g., -2024-08-06): Allows developers to lock in a specific version to ensure "Output Stability" and prevent software breakage during updates.

**3. The DeepSeek Series**

DeepSeek suffixes describe how "thinking" was achieved and how the architecture manages parameter efficiency.

Reasoning Core:

- -Zero: The "Raw RL" version. Trained without human examples, it "thinks out loud" in a raw, highly intelligent, yet sometimes chaotic way.
- -Distill: Identifies a "Student" model. A smaller architecture (like Llama or Qwen) taught to mimic the logic of the massive DeepSeek-R1 (685B).

Efficiency & Architecture:

- -MoE (Mixture of Experts): A "Sparse" model that only activates a fraction of its parameters (e.g., 21B out of 236B) per task for extreme speed.
- -MLA (Multi-Head Latent Attention): A 2026-standard suffix for massive memory savings in long conversations.
- -V2.5 / -V3: Mid-gen upgrades that merge specialized coding and chat branches into one unified weights file.

**4. The Qwen & Alibaba Series**

Qwen models are highly optimized for technical tasks, mathematics, and Eastern linguistic nuances.

Domain Specialists:

- -Coder: Specifically fine-tuned on vast repositories of code; optimized for software engineering.
- -Math: Optimized for solving complex equations and generating high-quality LaTeX output.

Generational Suffixes:

- -Qwen3: The latest 2026 architecture utilizing advanced attention mechanisms to reduce KV Cache memory requirements.

**5. The Phi & Gemma Collections**

Microsoft and Google focus on "Smarter, not Bigger," targeting high performance in compact sizes.

Phi & Gemma Types:

- Phi-4: Trained on "synthetic textbooks," allowing a 3B model to rival the logic of much larger 70B models.
- -it (Instruction Tuned): The Google/Gemma equivalent of "Instruct."
- -pt (Pre-trained): The Google/Gemma equivalent of "Base."

Framework Suffixes:
- -pytorch / -keras / -jax: Indicates which specific software library the model is formatted for.

**6. Quantization & Performance**

These suffixes describe how a model was "shrunk" to fit into specific VRAM or RAM limits.

Mathematical Precision:

- -BF16 / -FP16: Uses precision; high quality but requires maximum memory.
- -FP8 / -INT8: Uses 8-bit math to halve memory usage with negligible quality loss.
- -Q4_K_M / -int4: 4-bit quantization, shrinking the model by roughly 70-75 percent for consumer hardware.

Library & Method:

- -GGUF: Designed for llama.cpp and Ollama; optimized for CPU and Apple Silicon.
- -AWQ / -GPTQ: Standardized formats for high-speed NVIDIA GPU cloud deployment.
- -QAT (Quantization Aware Training): A premium version where the model was "born small," retaining higher intelligence than models shrunk after training.

**7. BERT & Encoder Suffixes:**

BERT models focus on "reading" and "understanding" rather than generating text.

- -cased vs. -uncased: "Cased" distinguishes between "Apple" (company) and "apple" (fruit); "Uncased" treats everything as lowercase for robustness.
- -WWM (Whole Word Masking): An advanced training technique where the model hides entire words rather than fragments, boosting context accuracy.

### Fine Tuning LLMs

Fine-tuning a Large Language Model (LLM) is the process of taking a "pre-trained" model (which has already learned general language, grammar, and facts from the internet) and training it further on a specific, smaller dataset.

**Types of Fine-Tuning**

- Supervised Fine-Tuning (SFT): This is the fundamental "bridge" from a base model to an assistant. It uses curated datasets of Prompt-Response pairs. The model learns to map specific input patterns to a structured output format, establishing the "personality" and baseline reliability of the AI.
- Instruction Tuning: A specialized form of SFT that uses high-diversity datasets (like Dolly or Alpaca) to teach the model how to follow commands. Instead of just "knowing facts," the model learns to execute tasks like summarization, brainstorming, or creative writing across a wide variety of topics.
- Self-Supervised Fine-Tuning (SSFT): In this method, the model acts as its own teacher. It is often used to improve reasoning capabilities. The model generates multiple solutions to a problem, evaluates them (sometimes using an external tool like a Python interpreter or a logical verifier), and then fine-tunes itself only on the successful "chains of thought." This is a key part of how "reasoning models" like DeepSeek-R1 are developed.
- Semi-Supervised Fine-Tuning: This approach combines a small amount of high-quality labeled data with a large amount of unlabeled data. The model uses the labeled data to learn the task format, then "pseudo-labels" the unlabeled data by predicting what the response should be. After filtering for high-confidence predictions, the model is trained on both the real and predicted labels.
- Domain Adaptation (Continual Pre-training): This involves training the model on a massive corpus of raw, unlabeled text from a specific field (e.g., legal case law or medical research). Unlike SFT, there are no questions or answers; the model simply learns the "language," vocabulary, and deep statistical patterns of that industry.
- Alignment (RLHF & DPO): These are "preference-based" types of tuning used after SFT to refine behavior.
    - RLHF (Reinforcement Learning from Human Feedback): Uses a separate "Reward Model" that mimics human judgment to score the LLM's answers, pushing it toward outputs that humans rank as more helpful.
    - DPO (Direct Preference Optimization): A more efficient alternative that directly optimizes the LLM to choose the "Better" answer over the "Worse" answer in a dataset of paired choices, skipping the need for a separate reward model.
- Task-Specific Fine-Tuning: The most narrow form of tuning, where the model is optimized for a single, repetitive output, such as converting natural language into SQL queries or extracting specific entities from medical prescriptions.
- Multimodal Fine-Tuning: A specialized type where a model is trained on pairs of different data types, such as Image-Text or Audio-Text. This allows the model to "see" or "hear" by learning to align visual/auditory tokens with language tokens.

**Ways to Fine-Tune LLMs**

There are several strategies for fine-tuning, ranging from updating the entire model to only touching a tiny fraction of it:
- Full Fine-Tuning : In this method, every single parameter in the model is updated. While it is highly accurate, it is extremely expensive in terms of VRAM and compute, often requiring industrial-grade hardware clusters. This method is best for massive domain shifts, such as teaching a general model to become a specialized legal or medical expert. However, it risks Catastrophic Forgetting, where the model loses its general reasoning abilities while trying to master the new task.
- Parameter-Efficient Fine-Tuning (PEFT) : These methods are the industry standard for most users because they are fast and resource-light. Instead of training the whole model, you only train a small "adapter" or a subset of layers.
    - LoRA (Low-Rank Adaptation): You inject small, trainable mathematical matrices into the Transformer layers. The original model weights remain frozen, and only these tiny matrices are updated.
    - QLoRA: A step further where the base model is quantized (compressed) to 4-bit, and LoRA is applied on top. This allows you to fine-tune a massive model (like Llama 70B) on a single consumer GPU.
    - IA3 (Infused Adapter by Inhibiting and Amplifying Inner Activations): A method that scales inner activations by learned vectors, achieving efficiency even greater than LoRA for certain tasks.
- Additive Tuning (Prompt Tuning and Prefix Tuning) : Instead of changing the model's weights, these methods focus on the input space.
    - Prompt Tuning: You keep the model frozen and learn a "soft prompt"—a series of continuous vectors that act as a virtual prefix to your input.
    - Prefix Tuning: Similar to prompt tuning, but it prepends trainable parameters to every layer of the transformer architecture, not just the input layer, providing more steering power.
- Adapter-Based Tuning : This involves inserting small bottleneck layers (adapters) between existing layers of the pre-trained model. Only these new layers are trained. It allows for "modular" AI, where you can swap different adapters onto the same base model to perform different tasks.
- Preference Alignment (RLHF and DPO) : Once a model has been fine-tuned on instructions, it often undergoes an alignment phase to match human preferences.
    - RLHF (Reinforcement Learning from Human Feedback): This uses a separate Reward Model to score LLM outputs, utilizing PPO (Proximal Policy Optimization) to adjust the LLM’s behavior toward safer or more helpful responses.
    - DPO (Direct Preference Optimization): A more recent alternative to RLHF that eliminates the need for a separate reward model. DPO directly optimizes the model using a binary cross-entropy loss on a dataset of "preferred" vs. "rejected" responses, making the alignment process more stable and computationally efficient.

**Fine Tune LLMs Implementation Framework**

When you are ready to write code, your choice of framework determines your level of control—from high-level automation to low-level tensor of LLMs manipulation.Choosing a framework depends on whether you want "push-button" simplicity or "from-scratch" control.
- Hugging Face Trainer API (The Industry Standard) The Trainer and SFTTrainer (from the TRL library) are the most popular ways to fine-tune. This method sits on top of PyTorch but automates the "boring" parts like manual gradient clipping, device placement, and logging. It is designed specifically for supervised fine-tuning and handles data formatting and "packing" (combining short samples) automatically to save GPU time.
- Native PyTorch (Maximum Control) This is the "manual" way to fine-tune, providing total control over the training process. You write the manual training loop, define your own optimizer (like AdamW), and handle backpropagation yourself. This is essential for researchers who are inventing new loss functions or need to implement custom sharding techniques like FSDP (Fully Sharded Data Parallel).
- Keras and TensorFlow (Modular Simplicity) Using Keras 3 or KerasHub, you can load models like Gemma or Llama as standard Keras layers. This allows you to use the familiar model.compile() and model.fit() syntax. It is a great choice for developers who want to integrate LLMs into a larger TensorFlow production pipeline or leverage TPU (Tensor Processing Unit) acceleration.
- Efficiency-First Frameworks (Unsloth & Axolotl)
    - Unsloth: A specialized wrapper that manually rewrites the mathematical kernels of the model. It can make fine-tuning up to 2x faster and use 70% less memory than standard PyTorch.
    - Axolotl: A configuration-based framework where you define your model and parameters in a simple .yaml file. It handles the backend complexities for you, making it excellent for reproducible and scalable training runs.
    - Llama-Factory: A comprehensive, user-friendly framework that offers a "no-code" web interface (Gradio) to manage the entire fine-tuning process.
    - DeepSpeed: A Microsoft library designed for high-end users who need to perform massive-scale full fine-tuning across many GPUs.

### HuggingFace API Parameters

#### Tokenizer Parameters

`AutoTokenizer.from_pretrained`

The tokenizer converts text into numbers. If configured incorrectly, the model will output gibberish.

`pretrained_model_name_or_path`
- Input: String (matches the model path on HuggingFace website).
- Function: Loads the specific vocabulary and merge rules for the model.
- Result: Ensures the numbers (tokens) sent to the model align with what it was originally trained on.

`use_fast`
- Input: Boolean (True, False).
- Function: Uses a Rust-based "Fast" tokenizer implementation.
- Result: Significantly speeds up the tokenization of large datasets. Almost always set to True.

`padding_side`
- Input: String ("left", "right").
- Function: Determines where the "padding" tokens are added to make sentences the same length.
- Result: For Causal LLMs (text generation), you must use "left" during inference, but "right" is often used for training with SFTTrainer.

`add_eos_token / add_bos_token`
- Input: Boolean (True, False).
- Function: Automatically adds "End of Sentence" or "Beginning of Sentence" markers.
- Result: Vital for fine-tuning. If the model doesn't learn when to stop (EOS), it will generate endless loops of text.

`truncation`
- Input: Boolean (True, False).
- Function: Cuts off sequences that are longer than the model's max context window.
- Result: Prevents memory errors when a specific document in your dataset is too long for the model to handle.

#### Model Loading Parameters

`AutoModelForCausalLM.from_pretrained`

These parameters dictate the "physical" state of the model on your hardware.

`pretrained_model_name_or_path`
- Input: String (e.g., "meta-llama/Llama-3-8b", "./local_model_folder/").
- Function: The unique identifier or path to the model weights.
- Result: Points the library to the specific architecture and weights to download or load.

`device_map`
- Input: String ("auto", "balanced") or Dict (mapping layers to specific GPUs).
- Function: Dictates which parts of the model go to which GPU/CPU.
- Result: Setting this to "auto" automatically handles "Model Parallelism," splitting the model across multiple GPUs if one isn't large enough.

`quantization_config`
- Input: BitsAndBytesConfig object.
- Function: Configures 4-bit or 8-bit loading (QLoRA).
- Result: Drastically reduces VRAM usage. For example, loading an 8B model in 4-bit requires ~5GB of VRAM instead of 16GB.

`torch_dtype`
- Input: torch.dtype (e.g., torch.float16, torch.bfloat16, "auto").
- Function: Sets the precision of the weights.
- Result: bfloat16 is preferred for modern GPUs to maintain numerical stability during fine-tuning.

`trust_remote_code`
- Input: Boolean (True, False).
- Function: Allows the execution of custom modeling code provided by the model author on the Hub.
- Result: Required for many new architectures (like Falcon or Phi) that haven't been merged into the core Transformers library yet.

`attn_implementation`
- Input: String ("eager", "sdpa", "flash_attention_2").
- Function: Chooses the mathematical engine for the Attention mechanism.
- Result: "flash_attention_2" is significantly faster and more memory-efficient but requires specific hardware (A100, H100, RTX 30/40 series).

`low_cpu_mem_usage`
- Input: Boolean (True, False).
- Function: Changes how the model is initialized to avoid using 2x the model size in CPU RAM.
- Result: Prevents system crashes (OOM) during the initial loading phase.

#### Fine Tuning Parameters

`TrainingArguments`

`TrainingArguments` defines the execution logic of the fine-tuning process. Because LLMs are computationally heavy, these parameters are the difference between a successful training run and prevent a "CUDA Out of Memory" (OOM) error. Here is a breakdown of the primary TrainingArguments grouped by their functional impact.

**1. Hardware & Memory Efficiency**

These parameters dictate how the model interacts with your GPU.

`per_device_train_batch_size`
- Input: Integer (e.g., 1, 4, 8).
- Function: The number of training examples processed simultaneously on a single GPU.
- Result: Higher values result in more stable gradient updates and faster training but require significantly more VRAM. For 7B models on 24GB GPUs, this is often set to 1 or 2.

`gradient_accumulation_steps`
- Input: Integer (e.g., 4, 8, 16).
- Function: The number of steps to accumulate gradients before performing a single weight update.
- Result: This allows you to simulate a large Global Batch Size without needing a massive GPU. (Effective Batch Size = per_device_train_batch_size × gradient_accumulation_steps × num_gpus).

`gradient_checkpointing`
- Input: Boolean (True, False).
- Function: Re-computes activations during the backward pass instead of storing them all in memory.
- Result: Reduces VRAM usage by 30-50% at the cost of roughly 20% slower training speed. Essential for fine-tuning models larger than 7B on consumer hardware.

`fp16 or bf16`
- Input: Boolean (True, False).
- Function: Enables Mixed Precision training. BF16 (BFloat16) is the 2026 industry standard for LLMs.
- Result: Halves memory usage and speeds up training. BF16 is preferred over FP16 because it has a wider dynamic range, preventing the "NaN" loss errors common in LLM fine-tuning.

**2. Learning & Optimization Logic**

These "knobs" control how the model learns from your specific dataset.

`learning_rate`
- Input: Float (e.g., 2e-4, 5e-5).
- Function: Controls the size of the steps the optimizer takes when updating weights.
- Result: High values (e.g., 1e-3) lead to fast but unstable learning and "catastrophic forgetting." Low values (e.g., 1e-6) are safer but may fail to adapt the model to your specific style or data.

`lr_scheduler_type`
- Input: String ("linear", "cosine", "constant", "cosine_with_restarts").
- Function: Defines how the learning rate changes over the course of training.
- Result: "cosine" is standard for smooth convergence; "linear" is often used for shorter instruction-tuning runs.

`warmup_ratio or warmup_steps`
- Input: Float (0.03) or Integer (100).
- Function: Gradually increases the learning rate from zero to the target rate at the start of training.
- Result: Prevents the model from being "shocked" by massive gradients in the first few batches, leading to much more stable convergence.

`weight_decay`
- Input: Float (e.g., 0.01, 0.1).
- Function: A penalty added to the loss function based on the size of the weights (Regularization).
- Result: Prevents the model from becoming too reliant on specific training examples (overfitting).

`optim`
- Input: String ("adamw_torch", "paged_adamw_32bit", "paged_adamw_8bit").
- Function: The algorithm used to update weights.
- Result: Using "paged_adamw_8bit" is the standard for QLoRA. It offloads optimizer states to CPU RAM when the GPU is full, effectively preventing OOM errors.

**3. Training Duration & Checkpointing**

These parameters manage the timeline of your fine-tuning run.

`num_train_epochs`
- Input: Float or Integer (e.g., 1, 3).
- Function: The number of times the model sees the entire dataset.
- Result: For LLM fine-tuning, 1 to 3 epochs is usually sufficient. More can lead to "overfitting," where the model mimics the data exactly but loses its general intelligence.

`max_steps`
- Input: Integer (e.g., 500, 1000).
- Function: Overrides epochs to stop training after a specific number of updates.
- Result: Useful for long datasets where you only want to perform a specific amount of training.

`save_steps / save_strategy`
- Input: Integer (100) / String ("steps", "epoch").
- Function: How often the model "saves" its progress to disk.
- Result: Frequent saving protects against hardware crashes but consumes significant disk space (a 7B model checkpoint is approx. 14GB-28GB).

**4. Logging & Evaluation**

Parameters for tracking the "health" of the training process.

`logging_steps`
- Input: Integer (e.g., 1, 10).
- Function: How often the loss (error rate) is printed to the console.
- Result: Lower values provide a smoother "Loss Curve" in tools like Weights & Biases (W&B).

`eval_strategy`
- Input: String ("steps", "epoch", "no").
- Function: Determines when to run the model against a "validation set" it hasn't seen yet.
- Result: Essential for verifying that the model is actually learning and not just memorizing.

`report_to`
- Input: String or List ("tensorboard", "wandb", "none").
- Function: Where to send the training metrics for visualization.

**5. Advanced Optimization & Stability**

These parameters ensure the training process doesn't "explode" and that the model updates its weights in a controlled manner.

`max_grad_norm`
- Input: Float (e.g., 0.3, 0.5, 1.0).
- Function: Clips the maximum value of the gradients (the steepness of the update).
- Result: This is vital for model stability, especially in Mixture-of-Experts (MoE) or 4-bit models. It prevents a single "bad" batch from causing massive weight shifts that would ruin the pre-trained knowledge.

`warmup_steps`
- Input: Integer (e.g., 100, 500).
- Function: The number of initial steps where the learning rate grows from zero to its maximum value.
- Result: Unlike warmup_ratio (which is a percentage of total training), this provides a fixed window of time for the model to "warm up" its adapters. It leads to much more stable loss curves at the very beginning of a run.

`ddp_find_unused_parameters`
- Input: Boolean (True, False).
- Function: Used in multi-GPU (Distributed Data Parallel) training to check for layers that aren't participating in the gradient calculation.
- Result: Setting this to False (as in your code) reduces overhead and speeds up training. It is almost always set to False when using LoRA because the adapters are the only things being updated.

**6. Model Selection & Checkpoint Management**

These parameters control which version of the model you actually keep and how much disk space you use.

`load_best_model_at_end`
- Input: Boolean (True, False).
- Function: At the end of the training, the trainer will automatically reload the weights from the version that performed best on your evaluation set.
- Result: Ensures that even if the model started "overfitting" in the final epoch, you walk away with the most generalized version of your fine-tune.

`metric_for_best_model`
- Input: String (e.g., "eval_loss", "accuracy").
- Function: Specifies the criteria used to decide which model version is the "best."
- Result: Usually set to eval_loss. It directs the trainer to prioritize the model that had the lowest error on data it wasn't trained on.

`greater_is_better`
- Input: Boolean (True, False).
- Function: Tells the trainer whether a higher value for your chosen metric is good (like Accuracy) or bad (like Loss).
- Result: For eval_loss, this must be False. This setting prevents logic errors where the trainer accidentally saves the worst-performing model.

`save_total_limit`
- Input: Integer (e.g., 2).
- Function: Limits the number of model checkpoints saved in your output_dir.
- Result: Crucial for disk management. Since each LLM checkpoint can be several gigabytes, this prevents your hard drive or cloud storage from filling up by deleting older, inferior checkpoints.

**7. Data Handling & Performance**

`remove_unused_columns`
- Input: Boolean (True, False).
- Function: Automatically drops columns from your dataset that aren't mentioned in the model's forward function.
- Result: Setting this to False is often necessary when using custom datasets or TRL’s SFTTrainer to ensure the "text" or "label" columns aren't accidentally deleted before the trainer can process them.

`dataloader_pin_memory`
- Input: Boolean (True, False).
- Function: Keeps the data in "pinned" (non-pageable) CPU memory to speed up transfers to the GPU.
- Result: While it can speed up data loading, setting it to False (as in your code) is a safer bet on systems with limited RAM to avoid system-wide lag or crashes.

`group_by_length`
- Input: Boolean (True, False).
- Function: Batches together sequences of similar lengths.
- Result: If False, the trainer uses random sampling. While True can speed up training by reducing padding, it is often disabled (False) for models with specific architectures like MoE (Phi-3.5 or Ministral) to maintain uniform distribution across the "experts."

**8. Logging & Identification**

`run_name`
- Input: String (e.g., "phi-3.5-moe-lora-qa").
- Function: A human-readable label for this specific training experiment.
- Result: Used to identify the run in logging dashboards like Weights & Biases or Tensorboard. This makes it much easier to compare different hyperparameter settings later.

`output_dir`
- Input: String/Path.
- Function: The folder where all logs, checkpoints, and the final model will be saved.
- Result: Organizing these by run names prevents you from accidentally overwriting a successful fine-tune with a new experiment.

**Special SFTTrainer Parameters (TRL Specific)**

Beyond the standard TrainingArguments, the TRL SFTTrainer uses these inputs:
- `packing` (Boolean): Combines multiple short examples into a single sequence.
    - Result: Increases training speed by 2x-4x by eliminating wasted space (padding) in your GPU memory.
- `max_seq_length` (Integer): The maximum context window for training (e.g., 2048, 4096).
    - Result: Directly impacts VRAM. Doubling this usually requires significantly more memory.
 
### Model Family Taxonomy

```
                        ┌───────────────────────────────────────┐
                        │           Transformer LM              │
                        └───────────────────────────────────────┘
                                       │
        ┌──────────────────────────────┼──────────────────────────────┐
        ▼                              ▼                              ▼
  Dense Decoder               Sparse (MoE)                   Hybrid (SSM + Attention)
  ├─ Pythia-410M              ├─ Phi-3.5-MoE (16 experts)     ├─ Jamba 2 (Mamba + Attn)
  ├─ GPT-2-medium             ├─ Phi-mini-MoE                 └─ Falcon-Mamba (ref)
  ├─ Llama-3.1-8B             ├─ Qwen1.5-MoE (64 experts)
  ├─ Qwen3-8B                 └─ DeepSeek-V2-Lite (MLA + MoE)
  ├─ Falcon-3-7B
  ├─ OLMo-2-7B
  └─ Phi-3.5-mini

  Reasoning-tuned (LRM)                      Small LM (SLM)
  ├─ Qwen3-4B-Thinking                       ├─ Phi-3.5-mini
  ├─ DeepSeek-R1-Distill-Qwen-1.5B           ├─ Gemma-3-1B
  ├─ Phi-4-mini-reasoning                    ├─ Ministral-3-3B
  ├─ Ministral-3-3B-Reasoning                └─ SmolLM3-3B
  └─ LFM2.5-1.2B-Thinking
```

**Dense Decoder** — every parameter is active for every token. Predictable scaling,
highest knowledge density.

**Mixture-of-Experts (MoE)** — a router selects the top-$k$ experts per token. Total
parameter count can be 10–50× the *active* parameter count, giving massive capacity
at the compute cost of a small model.

**Hybrid SSM** — interleave Mamba (linear state-space) layers with sparse attention
layers. Linear-time inference for 100k+ contexts while retaining global recall.

**Large Reasoning Model (LRM)** — trained with chain-of-thought RL. Uses
inference-time compute scaling: a small LRM that thinks for 20 seconds often beats a
larger dense model that answers in one second.

**Small Language Model (SLM)** — 100M–10B parameters, trained on heavily curated
"textbook-quality" data. Runs on phones/laptops.

---

## Models Surveyed

| Category | Model | Params | Active | Technique | Config |
|----------|-------|--------|--------|-----------|--------|
| **GPT (small)** | `EleutherAI/pythia-410m` | 405M | 405M | Full FT | `pythia_410m.yaml` |
| **GPT (small)** | `gpt2-medium` | 355M | 355M | Full FT | `gpt2_medium.yaml` |
| **GPT (large)** | `meta-llama/Llama-3.1-8B-Instruct` | 8B | 8B | QLoRA | `llama_3_1_8b.yaml` |
| **Qwen** | `Qwen/Qwen3-8B` | 8B | 8B | QLoRA | `qwen3_8b.yaml` |
| **MoE** | `microsoft/Phi-3.5-MoE-instruct` | 42B | 6.6B | BF16 LoRA | `phi_3_5_moe.yaml` |
| **SLM** | `microsoft/Phi-3.5-mini-instruct` | 3.8B | 3.8B | QLoRA | `phi_3_5_mini.yaml` |

Additional models implemented but not trained in the default pipeline (due to
compute constraints): DeepSeek-V2-Lite, Jamba2-3B, Qwen1.5-MoE, Qwen3-4B-Thinking,
DeepSeek-R1-Distill-Qwen-1.5B, LFM2.5-1.2B-Thinking, Phi-4-mini-reasoning,
Ministral-3-3B-Reasoning, Gemma-3-1B, SmolLM3-3B.

---

## Repository Structure

```
llm-finetuning-suite/
│
├── README.md                     ← this file
├── requirements.txt
├── setup.py
├── .gitignore
│
├── configs/                      # YAML configs (deep-merged with base_config.yaml)
│   ├── base_config.yaml
│   ├── pythia_410m.yaml
│   ├── gpt2_medium.yaml
│   ├── llama_3_1_8b.yaml
│   ├── qwen3_8b.yaml
│   ├── phi_3_5_moe.yaml
│   └── phi_3_5_mini.yaml
│
├── data/                         # Dataset CSVs (gitignored, downloaded at runtime)
│   └── .gitkeep
│
├── scripts/
│   └── download_data.py          # Kaggle downloader (requires kaggle.json)
│
├── src/                          # Importable Python package
│   ├── data/
│   │   ├── loader.py             # load_stacksample(), clean_html()
│   │   ├── preprocessing.py      # preprocess_qa_pairs(), split_dataset()
│   │   └── tokenization.py       # model-specific chat templates
│   ├── models/
│   │   ├── loader.py             # load_tokenizer(), load_model()
│   │   ├── quantization.py       # BitsAndBytesConfig builders
│   │   └── peft_config.py        # LoraConfig + apply_peft()
│   ├── training/
│   │   ├── trainer.py            # TrainingArguments builder + Trainer wrapper
│   │   └── callbacks.py          # EpochMonitor, EvaluationMonitor
│   ├── evaluation/
│   │   ├── metrics.py            # ROUGE, BLEU
│   │   ├── generation.py         # 5 generation strategies
│   │   └── evaluator.py          # perplexity + strategy comparison
│   └── utils/
│       ├── config.py             # YAML loader + numeric sanitizer
│       ├── monitoring.py         # ResourceMonitor, EpochMonitor
│       └── logging.py
│
├── notebooks/
│   ├── 01_fine_tuning_comparison.ipynb
│   └── 02_evaluation_comparison.ipynb
│
└── outputs/                      # Fine-tuned checkpoints (gitignored)
    └── .gitkeep
```

---

## Getting Start

**Prerequisites**

- **OS:** Linux or macOS (WSL2 works on Windows)
- **Python:** ≥ 3.10
- **CUDA:** ≥ 12.1 recommended (11.8 works)
- **GPU:** ≥ 16 GB VRAM for the 8B QLoRA experiments; ≥ 10 GB for everything else
- **Disk:** ≥ 30 GB free (dataset + checkpoints)

### Kaggle API Credentials

The StackSample dataset is hosted on Kaggle and requires an API token.

**Step 1 — Create a Kaggle account (if you don't have one)**
Go to <https://www.kaggle.com> and register.

**Step 2 — Generate an API token**
1. Click your avatar (top-right) → **Settings**
2. Scroll to **API** → click **Create New Token**
3. A file `kaggle.json` will be downloaded. It contains:

```json
{"username":"<your_username>","key":"<your_api_key>"}
```

**Step 3 — Place the token where the Kaggle CLI expects it**

```bash
mkdir -p ~/.kaggle
mv ~/Downloads/kaggle.json ~/.kaggle/kaggle.json
chmod 600 ~/.kaggle/kaggle.json
```

**Step 4 (alternative) — Use environment variables**

```bash
export KAGGLE_USERNAME="<your_username>"
export KAGGLE_KEY="<your_api_key>"
```

**Step 5 — Verify**

```bash
kaggle datasets list -s stacksample
```

If you see `stackoverflow/stacksample` in the output, you're ready.

### Environment Installation

```bash
# 1. Clone the repository
git clone https://github.com/<your-org>/a-survey-of-LLMs-architectures.git
cd a-survey-of-LLMs-architectures

# 2. Create an isolated environment (recommended)
python -m venv .venv
source .venv/bin/activate          # Linux / macOS
# .venv\Scripts\activate.bat       # Windows

# 3. Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# 4. Install the package in editable mode
pip install -e .

# 5. (Optional) Authenticate with Hugging Face for gated models
huggingface-cli login
#  → paste a token from https://huggingface.co/settings/tokens
#  → required for meta-llama/Llama-3.1-8B-Instruct
```

**Optional — Flash Attention 2** (only for 8B+ training runs):

```bash
pip install flash-attn --no-build-isolation
```

Requires an Ampere+ GPU (A100, H100, RTX 30/40 series).

### Data Preparation

**Option A — Download automatically**

```bash
python scripts/download_data.py
```

The script will:
1. Verify your Kaggle credentials.
2. Download `stacksample.zip` into `data/`.
3. Extract `Questions.csv`, `Answers.csv`, `Tags.csv`.
4. Print a file-size report.

Expected output:

```
==================================================
StackSample Dataset Downloader
==================================================
Kaggle credentials found at ~/.kaggle/kaggle.json
Downloading StackSample dataset...
Extracting stacksample.zip...
Dataset extracted to data/
All required files present:
    Questions.csv: 1.92 GB
    Answers.csv: 1.61 GB
    Tags.csv: 0.03 GB
Ready to use. Run notebooks/01_fine_tuning_comparison.ipynb
```

**Option B — Download manually**

```bash
mkdir -p data
kaggle datasets download -d stackoverflow/stacksample -p data
unzip data/stacksample.zip -d data/
```

**Option C — Use a temporary working copy**

If your dataset lives elsewhere (e.g. a mounted SSD), just point `data_dir`
at it when calling `load_stacksample()`:

```python
from src.data.loader import load_stacksample

questions, answers = load_stacksample(
    data_dir="/mnt/ssd/stacksample",
    score_threshold=5,
    max_questions=100,
)
```

> **Note on memory.** `Answers.csv` is 1.61 GB; `Questions.csv` is 1.92 GB. The
> loader uses Polars (lazy CSV parsing) and immediately filters down to the top-100
> highest-scoring questions before any HTML cleaning. Peak RAM usage is typically
> under 4 GB.

### Running Experiments

#### Experiment 1 — Fine-Tuning Comparison

**Notebook:** `notebooks/01_fine_tuning_comparison.ipynb`

This notebook walks through:

1. Loading and splitting the StackSample Q&A corpus (80/14/6 train/test/val by *unique
   question title*).
2. For each of the 6 default configs, applying the correct tokenizer chat template.
3. Loading the model with the right quantization + PEFT strategy.
4. Training with `TrainingArguments` built from the YAML config.
5. Producing a **per-epoch table** with wall-clock time, average CPU, and average
   memory. The result is a single comparable summary for all six models.

**Run it:**

```bash
jupyter notebook notebooks/01_fine_tuning_comparison.ipynb
```

Then execute the cells top-to-bottom. On a single RTX 4090:

| Model | Approx. epoch time | VRAM peak |
|-------|-------------------|-----------|
| Pythia-410M (Full FT) | 18 s | ~4 GB |
| GPT-2-medium (Full FT) | 36 s | ~4 GB |
| Llama-3.1-8B (QLoRA) | 198 s | ~11 GB |
| Qwen3-8B (QLoRA) | 198 s | ~11 GB |
| Phi-3.5-MoE (BF16 LoRA) | — | ~80 GB (A100/H100 only) |
| Phi-3.5-mini (QLoRA) | 214 s | ~7 GB |

#### Experiment 2 — Evaluation Comparison

**Notebook:** `notebooks/02_evaluation_comparison.ipynb`

For every fine-tuned model, this notebook computes:

- **Perplexity** on the test split (intrinsic language-modelling quality).
- **ROUGE-1 / ROUGE-2 / ROUGE-L** against the human reference answers.
- **BLEU** (precision-oriented n-gram overlap).
- **Five generation strategies** — greedy, beam search, top-k, top-p,
  temperature sampling — each with its own ROUGE/BLEU/time profile.
- **Resource usage** — average seconds per query, CPU%, memory%.

Output is a unified comparison table so you can see, for example, that
**greedy search dominates ROUGE** on the distil-Qwen model but **beam search wins
on BLEU** for the reasoning-tuned Phi model.

**Run it:**

```bash
jupyter notebook notebooks/02_evaluation_comparison.ipynb
```

#### Running a single model from the command line

```python
import sys
sys.path.insert(0, "src")

from data.loader import load_stacksample
from data.preprocessing import preprocess_qa_pairs, split_dataset
from data.tokenization import build_dataset_dict
from models.loader import load_tokenizer, load_model
from models.quantization import build_quant_config
from models.peft_config import apply_peft
from training.trainer import run_training, print_training_results
from utils.config import load_config

cfg = load_config("configs/pythia_410m.yaml")

# 1. Data
q, a = load_stacksample(data_dir="data")
qa = preprocess_qa_pairs(q, a)
train_df, test_df, val_df = split_dataset(qa)

# 2. Tokenizer + dataset
tok = load_tokenizer(cfg["model"]["name"])
ds = build_dataset_dict(train_df, val_df, tok, cfg["model"]["name"])

# 3. Model
model = load_model(
    cfg["model"]["name"],
    quantization_config=build_quant_config(cfg.get("quantization")),
)
if cfg["model"].get("use_peft"):
    model = apply_peft(model, cfg["peft"])

# 4. Train
trainer, monitor, result = run_training(model, tok, ds, cfg)
print_training_results(trainer, monitor, result)
```

### Configuration Reference

Every experiment is fully specified by a YAML file that is **deep-merged** with
`configs/base_config.yaml`. You only override the fields you need.

```yaml
# configs/pythia_410m.yaml
model:
  name: "EleutherAI/pythia-410m"
  output_dir: "./outputs/pythia-410m-finetuned"
  use_peft: false
  use_quantization: false
  trust_remote_code: false

training:
  num_train_epochs: 3
  per_device_train_batch_size: 4
  gradient_accumulation_steps: 4
  learning_rate: 1.0e-4          # ← always use a decimal point
  optim: "adamw_torch"
  max_grad_norm: 1.0
  save_strategy: "no"
  load_best_model_at_end: false
```

### Numeric parsing gotcha

PyYAML parses `1e-4` (no decimal point) as a **string**, not a float. `src/utils/config.py`
silently coerces these back to floats after loading, so both `1.0e-4` and `1e-4` work
— but it's good hygiene to always write the decimal point.

### Full schema

| Section | Field | Purpose |
|---------|-------|---------|
| `model` | `name` | Hugging Face repo id |
| `model` | `output_dir` | Where the fine-tuned weights are saved |
| `model` | `use_peft` | Apply LoRA / QLoRA adapters |
| `model` | `use_quantization` | Enable 4-bit QLoRA |
| `model` | `trust_remote_code` | Required for DeepSeek, Qwen, etc. |
| `model` | `attn_implementation` | `"sdpa"` \| `"flash_attention_2"` \| `"eager"` |
| `quantization` | `load_in_4bit` | Enable 4-bit NF4 |
| `quantization` | `bnb_4bit_quant_type` | `"nf4"` (default) or `"fp4"` |
| `quantization` | `bnb_4bit_compute_dtype` | `"bfloat16"` (recommended) |
| `quantization` | `bnb_4bit_use_double_quant` | Extra ~0.4 bits/param savings |
| `peft` | `r` | LoRA rank (8–64 typical) |
| `peft` | `lora_alpha` | Usually `2 × r` |
| `peft` | `target_modules` | Which linear layers get adapters |
| `peft` | `lora_dropout` | Regularization |
| `training` | `learning_rate` | Step size; 1e-4 for small, 2e-5 for QLoRA |
| `training` | `num_train_epochs` | 1–3 for LLM fine-tuning |
| `training` | `optim` | `"paged_adamw_8bit"` for QLoRA |
| `training` | `max_grad_norm` | Gradient clipping; 0.3 for MoE, 0.5–1.0 for dense |
| `training` | `bf16` | Always `true` on modern GPUs |
| `training` | `gradient_checkpointing` | Trades speed for memory |

## Results and Metrics

Fine-tuning results (StackSample Q&A, 3 epochs, RTX A4000/A100):

| Model | Technique | Epoch time | Final train loss | Val loss trajectory |
|-------|-----------|------------|------------------|---------------------|
| Pythia-410M | Full FT | 18.4 s | 1.77 | 2.60 → 3.20 → 3.08 |
| GPT-2-medium | Full FT | 35.9 s | 2.08 | 2.79 → 2.88 → 2.92 |
| Llama-3.1-8B | QLoRA (r=64) | 197.9 s | 1.68 | 2.06 → 2.02 → 2.05 |
| Qwen3-8B | QLoRA (r=64) | 197.6 s | 1.73 | 2.15 → 2.15 → 2.25 |
| Phi-3.5-MoE | BF16 LoRA (r=16, q/v) | — | — | (requires ≥ 80 GB) |
| Phi-3.5-mini | QLoRA (r=16) | 213.9 s | 9.17 | 9.58 → 8.88 → 8.76 |

Evaluation results (greedy decoding unless noted):

| Model | Perplexity | ROUGE-1 | ROUGE-L | BLEU |
|-------|------------|---------|---------|------|
| DeepSeek-R1-Distill-Qwen-1.5B | 11.72 | 0.227 | 0.141 | 0.019 |
| Qwen3-4B-Thinking | 5.40 | 0.223 | 0.154 | 0.015 |

Generation-strategy comparison (DeepSeek-R1-Distill-Qwen-1.5B):

| Strategy | ROUGE-1 | ROUGE-L | BLEU | Avg (s) |
|----------|---------|---------|------|---------|
| **Greedy** | **0.286** | **0.168** | **0.035** | 4.37 |
| Beam search (4) | 0.273 | 0.168 | 0.013 | 4.37 |
| Top-k (50) | 0.207 | 0.125 | 0.009 | 4.24 |
| Top-p (0.9) | 0.223 | 0.140 | 0.007 | 4.25 |
| Temperature (0.8) | 0.198 | 0.129 | 0.006 | 4.19 |

**Interpretation:** For factual Q&A, greedy decoding is the single best strategy
across every overlap metric. Sampling methods add diversity at the cost of
relevance — exactly the opposite of what you'd want for storytelling tasks.

### Technical Reference

**The `src/` package at a glance**

| Module | Public functions |
|--------|------------------|
| `src.data.loader` | `load_stacksample`, `clean_html` |
| `src.data.preprocessing` | `preprocess_qa_pairs`, `split_dataset` |
| `src.data.tokenization` | `get_formatter`, `tokenize_dataset`, `build_dataset_dict` |
| `src.models.loader` | `load_tokenizer`, `load_model` |
| `src.models.quantization` | `build_quant_config` |
| `src.models.peft_config` | `build_lora_config`, `apply_peft` |
| `src.training.trainer` | `build_training_args`, `run_training`, `print_training_results` |
| `src.training.callbacks` | `EpochMonitor`, `EvaluationMonitor` |
| `src.evaluation.metrics` | `build_rouge_scorer`, `compute_rouge`, `compute_bleu` |
| `src.evaluation.generation` | `generate_answer`, `STRATEGY_CONFIGS` |
| `src.evaluation.evaluator` | `evaluate_model`, `calculate_perplexity` |
| `src.utils.config` | `load_config` |
| `src.utils.monitoring` | `ResourceMonitor`, `EpochMonitor` |

**Data dictionary**

**`Questions.csv`** (~1.26M rows)

| Column | Type | Notes |
|--------|------|-------|
| `Id` | int | 80 – 40M |
| `OwnerUserId` | int | 631k unique |
| `CreationDate` | datetime | 2008-08 → 2016-10 |
| `ClosedDate` | datetime | ~96% NA |
| `Score` | int | mean 1.78, max 5190 |
| `Title` | str | short headline |
| `Body` | str (HTML) | full description |

**`Answers.csv`** (~2.01M rows)

| Column | Type | Notes |
|--------|------|-------|
| `Id` | int | 92 – 40M |
| `OwnerUserId` | int | 469k unique |
| `CreationDate` | datetime | 2008-08 → 2016-10 |
| `ParentId` | int | FK → `Questions.Id` |
| `Score` | int | mean 2.48, max 5718 |
| `Body` | str (HTML) | full answer text |

**Hyperparameter defaults in `base_config.yaml`**

| Field | Default | Rationale |
|-------|---------|-----------|
| `max_length` | 128 | Fits in 4 GB VRAM with batch=4 on 3B+ models |
| `test_size` | 0.2 | 14% of unique questions |
| `val_size` | 0.3 | 6% of unique questions |
| `score_threshold` | 5 | Filters out low-quality answers |
| `max_questions` | 100 | Keeps the full run under ~1 hour on a 4090 |
| `per_device_train_batch_size` | 4 | Tuned for 24 GB GPUs |
| `learning_rate` | 2.0e-5 | Safe QLoRA default |
| `warmup_steps` | 100 | ~10% of typical run |
| `lr_scheduler_type` | `"cosine"` | Smooth convergence |

---

## Citation and References

If you use this repository in academic work, please cite the underlying papers.

### Core architectures

```bibtex
@inproceedings{vaswani2017attention,
  title     = {Attention Is All You Need},
  author    = {Vaswani, Ashish and others},
  booktitle = {NeurIPS},
  year      = {2017}
}

@article{su2024roformer,
  title   = {RoFormer: Enhanced Transformer with Rotary Position Embedding},
  author  = {Su, Jianlin and others},
  journal = {Neurocomputing},
  year    = {2024}
}

@article{shazeer2020glu,
  title   = {GLU Variants Improve Transformer},
  author  = {Shazeer, Noam},
  journal = {arXiv:2002.05202},
  year    = {2020}
}

@article{zhang2019rmsnorm,
  title   = {Root Mean Square Layer Normalization},
  author  = {Zhang, Biao and Sennrich, Rico},
  journal = {NeurIPS},
  year    = {2019}
}
```

### Optimization & parameter-efficient fine-tuning

```bibtex
@article{hu2021lora,
  title   = {LoRA: Low-Rank Adaptation of Large Language Models},
  author  = {Hu, Edward J. and others},
  journal = {ICLR},
  year    = {2022}
}

@article{dettmers2023qlora,
  title   = {QLoRA: Efficient Finetuning of Quantized LLMs},
  author  = {Dettmers, Tim and others},
  journal = {NeurIPS},
  year    = {2023}
}

@article{dao2022flashattention,
  title   = {FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness},
  author  = {Dao, Tri and others},
  journal = {NeurIPS},
  year    = {2022}
}
```

### Mixture-of-Experts and hybrid architectures

```bibtex
@article{fedus2022switch,
  title   = {Switch Transformers: Scaling to Trillion Parameter Models},
  author  = {Fedus, William and others},
  journal = {JMLR},
  year    = {2022}
}

@article{deepseek2024v2,
  title   = {DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts Language Model},
  author  = {{DeepSeek-AI}},
  journal = {arXiv:2405.04434},
  year    = {2024}
}

@article{gu2023mamba,
  title   = {Mamba: Linear-Time Sequence Modeling with Selective State Spaces},
  author  = {Gu, Albert and Dao, Tri},
  journal = {arXiv:2312.00752},
  year    = {2023}
}
```

### Reasoning and reinforcement learning

```bibtex
@article{deepseek2025r1,
  title   = {DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning},
  author  = {{DeepSeek-AI}},
  journal = {arXiv:2501.12948},
  year    = {2025}
}
```

### Datasets and benchmarks

```bibtex
@misc{stackoverflow2016stacksample,
  title  = {StackSample: 10\% of Stack Overflow Q\&A},
  author = {{Stack Overflow}},
  year   = {2016},
  url    = {https://www.kaggle.com/datasets/stackoverflow/stacksample}
}

@inproceedings{lin2004rouge,
  title     = {ROUGE: A Package for Automatic Evaluation of Summaries},
  author    = {Lin, Chin-Yew},
  booktitle = {Text Summarization Branches Out},
  year      = {2004}
}

@inproceedings{papineni2002bleu,
  title     = {BLEU: a Method for Automatic Evaluation of Machine Translation},
  author    = {Papineni, Kishore and others},
  booktitle = {ACL},
  year      = {2002}
}
```

### Model cards (Hugging Face)

| Model | URL |
|-------|-----|
| `EleutherAI/pythia-410m` | <https://huggingface.co/EleutherAI/pythia-410m> |
| `gpt2-medium` | <https://huggingface.co/openai-community/gpt2-medium> |
| `meta-llama/Llama-3.1-8B-Instruct` | <https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct> |
| `Qwen/Qwen3-8B` | <https://huggingface.co/Qwen/Qwen3-8B> |
| `Qwen/Qwen3-4B-Thinking-2507` | <https://huggingface.co/Qwen/Qwen3-4B-Thinking-2507> |
| `microsoft/Phi-3.5-MoE-instruct` | <https://huggingface.co/microsoft/Phi-3.5-MoE-instruct> |
| `microsoft/Phi-3.5-mini-instruct` | <https://huggingface.co/microsoft/Phi-3.5-mini-instruct> |
| `microsoft/Phi-4-mini-reasoning` | <https://huggingface.co/microsoft/Phi-4-mini-reasoning> |
| `microsoft/Phi-mini-MoE-instruct` | <https://huggingface.co/microsoft/Phi-mini-MoE-instruct> |
| `deepseek-ai/DeepSeek-V2-Lite` | <https://huggingface.co/deepseek-ai/DeepSeek-V2-Lite> |
| `deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B` | <https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B> |
| `tiiuae/Falcon3-7B-Instruct` | <https://huggingface.co/tiiuae/Falcon3-7B-Instruct> |
| `allenai/OLMo-2-1124-7B-Instruct` | <https://huggingface.co/allenai/OLMo-2-1124-7B-Instruct> |
| `mistralai/Ministral-3-3B-Instruct-2512-BF16` | <https://huggingface.co/mistralai/Ministral-3-3B-Instruct-2512-BF16> |
| `mistralai/Ministral-3-3B-Reasoning-2512` | <https://huggingface.co/mistralai/Ministral-3-3B-Reasoning-2512> |
| `google/gemma-3-1b-it` | <https://huggingface.co/google/gemma-3-1b-it> |
| `HuggingFaceTB/SmolLM3-3B` | <https://huggingface.co/HuggingFaceTB/SmolLM3-3B> |
| `ai21labs/AI21-Jamba2-3B` | <https://huggingface.co/ai21labs/AI21-Jamba2-3B> |
| `Qwen/Qwen1.5-MoE-A2.7B` | <https://huggingface.co/Qwen/Qwen1.5-MoE-A2.7B> |
| `LiquidAI/LFM2.5-1.2B-Thinking` | <https://huggingface.co/LiquidAI/LFM2.5-1.2B-Thinking> |

---

## License

MIT © the authors. See `LICENSE` for the full text.

The StackSample dataset is licensed by Stack Overflow under CC-BY-SA 3.0. Individual
model weights retain their original licenses (Llama 3.1 Community License, Apache 2.0,
Gemma Terms of Use, etc.) — check each model card before redistribution.