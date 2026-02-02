# Generative AI - Complete Guide

### What is Generative AI?

**Definition**: AI that creates new content (text, images, code, music, etc.)

**Key Difference**:
- **Discriminative AI**: Classifies/predicts (is this spam?)
- **Generative AI**: Creates new content (write an email)

**Analogy**:
- **Discriminative**: Art critic (judges existing art)
- **Generative**: Artist (creates new art)

### Types of Generative Models

#### 1. **Generative Adversarial Networks (GANs)**

**Concept**: Two networks competing

**Players**:
- **Generator**: Creates fake data (like counterfeiter)
- **Discriminator**: Detects fakes (like police)

**Training Process**:
```
1. Generator creates fake image
2. Discriminator tries to detect it's fake
3. Generator improves to fool discriminator
4. Discriminator improves to catch fakes
5. Repeat until generator creates realistic images
```

**Use Cases**:
- Image generation (StyleGAN)
- Face generation (This Person Does Not Exist)
- Art creation
- Data augmentation

**Limitations**:
- Training instability
- Mode collapse (generates similar outputs)

#### 2. **Variational Autoencoders (VAEs)**

**Concept**: Learn compressed representation, generate from it

**How it Works**:
```
Input → Encoder → Latent Space (compressed) → Decoder → Output
```

**Key Feature**: Latent space is continuous (can interpolate)
- Generate variations by moving in latent space

**Use Cases**:
- Image generation
- Anomaly detection
- Data compression

#### 3. **Diffusion Models** (Current State-of-the-Art)

**Concept**: Learn to reverse noise process

**Training**:
1. Start with real image
2. Add noise gradually (like static on TV)
3. Model learns to remove noise (denoise)
4. Can generate by starting with pure noise and denoising

**Why Popular**:
- More stable than GANs
- High quality outputs
- Used in DALL-E 2, Midjourney, Stable Diffusion

**Process**:
```
Pure Noise → Denoise Step 1 → Denoise Step 2 → ... → Final Image
```

#### 4. **Autoregressive Models** (Language Models)

**Concept**: Generate one token at a time, conditioned on previous

**How**:
- Predict next word given previous words
- "The cat sat" → predict "on"
- "The cat sat on" → predict "the"
- Continue until complete

**Examples**: GPT, PaLM, Claude

### Text Generation Models

#### GPT Series Evolution

**GPT-1** (2018):
- 117M parameters
- Proof of concept

**GPT-2** (2019):
- 1.5B parameters
- Showed scaling works
- Initially not released (fear of misuse)

**GPT-3** (2020):
- 175B parameters
- Few-shot learning (learn from examples in prompt)
- In-context learning

**GPT-4** (2023):
- Larger, multimodal (text + images)
- Better reasoning
- More reliable

**Key Concepts**:

**1. Prompt Engineering**
- Art of crafting inputs to get desired outputs
- **Few-shot**: Provide examples in prompt
- **Zero-shot**: No examples, just instruction
- **Chain-of-Thought**: Ask model to think step-by-step

**2. Temperature**
- Controls randomness
- Low (0.1): Deterministic, focused
- High (1.0): Creative, diverse
- **Analogy**: Conservative vs creative writer

**3. Top-k / Top-p Sampling**
- Limits choices to most likely tokens
- Prevents nonsensical outputs

**4. Fine-tuning**
- Adapt pre-trained model to specific task
- **Example**: Fine-tune GPT for code generation

### Image Generation Models

#### DALL-E (OpenAI)
- Text-to-image
- Uses CLIP + GPT-3
- Can edit images

#### DALL-E 2
- Improved quality
- Better text understanding
- Inpainting, outpainting

#### Midjourney
- Artistic style
- High aesthetic quality
- Popular for art generation

#### Stable Diffusion
- Open-source
- Runs on consumer GPUs
- Highly customizable
- Community-driven

#### Imagen (Google)
- High quality
- Better text rendering
- Not publicly available

### Code Generation

#### GitHub Copilot
- Powered by Codex (GPT-3 fine-tuned on code)
- Autocomplete, function generation
- Multi-language support

#### CodeT5, CodeBERT
- Specialized code models
- Code understanding and generation

### Video & Audio Generation

**Video**:
- RunwayML, Pika, Stable Video Diffusion
- Text-to-video, image-to-video

**Audio**:
- MusicLM (Google): Text-to-music
- AudioLM: Audio generation
- Voice cloning: ElevenLabs, Resemble

### Multimodal Models

**What**: Handle multiple types of input/output

**Examples**:
- **GPT-4 Vision**: Text + Images → Text
- **CLIP**: Understands images and text together
- **Flamingo**: Few-shot learning with images
- **PaLM-E**: Robotics + vision + language

### Challenges in Generative AI

**1. Hallucination**
- Model generates plausible but false information
- **Solution**: Fact-checking, retrieval-augmented generation

**2. Bias**
- Reflects biases in training data
- **Solution**: Diverse training data, bias mitigation

**3. Safety & Misuse**
- Deepfakes, misinformation
- **Solution**: Watermarking, content moderation

**4. Evaluation**
- Hard to measure quality objectively
- **Solution**: Human evaluation, automated metrics

### Prompt Engineering Best Practices

**1. Be Specific**
- ❌ "Write about AI"
- ✅ "Write a 500-word article about the impact of GPT-4 on software development, targeting developers"

**2. Provide Context**
- Include relevant information
- Set the tone and style

**3. Use Examples (Few-shot)**
- Show desired format
- Demonstrate patterns

**4. Chain-of-Thought**
- Ask for step-by-step reasoning
- Improves accuracy for complex tasks

**5. Iterate**
- Refine prompts based on outputs
- A/B test different phrasings

---

