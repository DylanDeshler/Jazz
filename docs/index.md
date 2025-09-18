# Introduction
The inspiration for this was the original [Genie paper](https://arxiv.org/pdf/2402.15391) from Google DeepMind. If you haven't read it or seen [Genie3](https://deepmind.google/discover/blog/genie-3-a-new-frontier-for-world-models/) I highly recommend checking it out. The basic premis of these models is controllable generation without supervised labels, because collecting these labels (especially at fine granularity) can be quite expensive. What they do instead is train a Latent Action Model (LAM) to predict what action occurs between frames. Then they use these unsupervised labels for conditional generation. The main dataset used in the paper is a large collection of platformer videos collected from YouTube. They restrict the action space to 8 actions which they find produces actions for simple movement. A great idea that according to the paper both works and scales well!

## Problem Statement
I want to take the Genie approach and apply it to music. But instead of focusing on 'live' playability, to focus on semantic editing. I do this for two main reasons:
  1. There is a ton of backlash (understandably so) from artistic communnities on AI generated art. While I am participating in that from the AI side of things, I want my work to be interesting and acceptable to both sides.
  2. Editing (aka in-painting) should be a simpler task than pure generation (ignoring boundary conditions). And because I am quite data and compute limited, simpler tasks are preferable.

There are many high quality music generation models that let someone generate music from text, and if we go back far enough in time, even from sheet music. But at the moment text doesn't have sufficient fine grained control, and I'm not yet convinced its even the right medium for that level of control. Instead I propose to learn a set of actions for that very purpose, and most of the fun along the way will be discovering what those actions are. 

## Overview
Conceptually I follow the same approach as the initial Genie paper. I train an audio tokenizer, a latent action model, and finally a dynamics model. However each step in my training pipeline has some crucial deviations from Genie. I'll go through them one by one and explain what motivates them and the results.

## Data
I found an incredible dataset of Jazz recordings called [JazzSet](https://www.reddit.com/r/datasets/comments/1b73vz3/jazzset_large_audio_dataset_with_instrumentation/?utm_source=chatgpt.com) on Reddit. The recordings date from the late 19th centrury to the early 21st and vary substantially in quality. Data quality is curcial in training a strong model so I built a filtering pipeline to retain the highest quality recordings from the dataset. It's important to note that rejecting all "old" songs would have done a somewhat reasonable job of improving the overall audio quality, but at a great cost to data coverage. Instead I trained a simple model to help with filtering.

## Data Filtering
I resampled all of the recordings to 16000hz to reduce the computational training burden while maintaining reasonable peak audio quality. I randomly sampled ~200 songs, took the first 5 seconds from each, and labeled the quality as good or bad. The sampling was biased by age of the recording, anything somewhat recent had high quality so I didn't want to fill my training set with good quality recordings and bias the model accordingly. I trained an ExtraTrees model on simple statistics calculated over frequency bands from the Mel Spectrogram of the 5 second cuts. After training the model had ~80% accuracy which I deemed good enough for filtering. To filter, I calculated the same feature set over the entire dataset, predicted the probability of each 5 second segment being good, applied a gaussian filter to the probabilities, and passed a song through if the average probability was above 55%. I decided to pass/fail entire songs instead of their segments because I didn't want to deal with stitching segments together (despite this being common in NLP pre-training) and I didn't know what the final generative models context length would be (i.e. how much stitching would impact training). In hindsight maintaining sufficiently long (>30s) clips that were high quality likely would have increased my dataset size without too much overhead.

Here is an example of the difference in audio quality, the crackaling is a characateristic of old recording equipment and should be excluded from the training set.
| Good | Bad |
|------|-----|
| <audio controls><source src="samples/mel/JV-36144-1957-QmbSPzr8VX8LUrnatKVGRK9G9wZuVxah5VdMBgXVTpNBDn.wav-TS485813.wav" type="audio/wav"></audio> | <audio controls><source src="samples/mel/JV-12-1916-QmYZJNDBn5WPRNakbBi9UujN8dMrJo3n8vpaHo8RNon4xt.wav-JV-12-1916-QmYZJNDBn5WPRNakbBi9UujN8dMrJo3n8vpaHo8RNon4xt.wav" type="audio/wav"></audio> |

Here are the Mel Spectrograms for the first 5 seconds of each song that were used to generate the features for the quality classifier.
| ![Good](samples/mel/good.png) | ![Bad](samples/mel/bad.png) |
|-----------------|-----------------|


# Audio Tokenizer
## Architectures
From a conceptual POV I am anti quantization of continuous signals in tokenizers, why are we restricting the continuous nature of our data to some dumb integers in high dimensional space?? Why not use that continuous structure to our advantage?? He says yelling at no one in particular. An annoying looking man begins to say something about digital sensors being discrete by conception before he is pumbled by the audience, everyone applauds. So anyway I just don't like them, but they are the most performant tokenizers, so I understand why they are commonly used. At the onset of this project I spent some time fooling around with pretrained tokenziers and trying to build my own continuous tokenizers. It took a fair few failed attempts and bad results before I realized the main issue was that I was bad at googling and the solution already existed on ArXiv. Boom! [Diffusion Tokenizers](https://arxiv.org/pdf/2501.18593v1) why predict the entire latent space in one go when you can do it iteratively instead? A simple and profound idea that happened to be what I was messing with, just better and with many of the kinks worked out. I adjusted the architecture to work with 1D audio signals instead of 2D images, which mostly involved baking in the recent advances in audio tokenizers, and it pretty much worked out of the box! I love it when that happens. I highly recommend reading the Diffusion Tokenizers paper because it requires no specialized losses. Yes you heard me right, no collection of STFT losses, no GAN losses, no LIPSIS losses, or anything domain specific. Simply a diffusion objective that scales better than anyother image tokenizer. This worked quite well for low compression (16000hz -> 50 tokens). Below are some sample reconstructions.

| Original | Reconstruction |
|----------|----------------|
| <audio controls><source src="samples/low/0_real.wav" type="audio/wav"></audio> | <audio controls><source src="samples/low/0_recon.wav" type="audio/wav"></audio> |
| <audio controls><source src="samples/low/1_real.wav" type="audio/wav"></audio> | <audio controls><source src="samples/low/1_recon.wav" type="audio/wav"></audio> |
| <audio controls><source src="samples/low/2_real.wav" type="audio/wav"></audio> | <audio controls><source src="samples/low/2_recon.wav" type="audio/wav"></audio> |
| <audio controls><source src="samples/low/3_real.wav" type="audio/wav"></audio> | <audio controls><source src="samples/low/3_recon.wav" type="audio/wav"></audio> |
| <audio controls><source src="samples/low/5_real.wav" type="audio/wav"></audio> | <audio controls><source src="samples/low/5_recon.wav" type="audio/wav"></audio> |

Unfortunately naively extending this approach to higher levels of compression failed miserably. Training a hierarchical tokenizer on top of the lower level tokenized latents also failed. [DC-AE](https://arxiv.org/pdf/2410.10733) to the rescue! The most insightful point in this paper is the realization that tokenizers have competing optimization processes that make learning difficult.
  Process 1: Move information from the sequence to channels, which is a difficult task for convolutional filters (large gradients)
  Process 2: Compress salient information and ditch noise (small gradients)
The different gradient magnitudes is an explanation for why adding additional blocks to enocders does not simply give better compression. The gradient signal is dominated by moving information around rather than better compression. DC-AE proposes to solve this problem by handling Process 1 with [Pixel Shuffle](https://docs.pytorch.org/docs/stable/generated/torch.nn.PixelShuffle.html), a technique that has been around a few years for better upsampling. There are no learned parameters in this process, so the large gradients disappear. Then the compression is handled by a similar but learned residual transformation with small gradients. With this new approach the model only needs to learn what information to add rather than how to move it around to the further compressed representation. Below are some sample reconstructions for the high level tokenizer (16000hz -> 8 tokens). Notice that the musical themes are consitent between the original and reconstruction but that the audio quality varies. This is because the tokenizer is too small and it is early in training.

| Original | Reconstruction |
|----------|----------------|
| <audio controls><source src="samples/high/2_real.wav" type="audio/wav"></audio> | <audio controls><source src="samples/high/2_recon.wav" type="audio/wav"></audio> |
| <audio controls><source src="samples/high/3_real.wav" type="audio/wav"></audio> | <audio controls><source src="samples/high/3_recon.wav" type="audio/wav"></audio> |
| <audio controls><source src="samples/high/7_real.wav" type="audio/wav"></audio> | <audio controls><source src="samples/high/7_recon.wav" type="audio/wav"></audio> |

To force the model to learn higher level and more longterm features, I doubled the inpute sequence length to 2 seconds and then later increased it to 5 seconds. This was faster and more computationally efficient for my setup than starting training with 5 second segments. Scaling up the model and training will improve audio quality which would likely improve downstream learning. However, I have limitied compute and time, and these reconstructions maintain the most salient information which is what is required for the next stage.

## Latent Shape
Taking advantage of the architectural modifications from DC-AE makes trading compression for latent dimension more tractable. The next question becomes what latent dimension has the best trade-off between compression and generation. Lets define what I mean by each of those terms in this specific context.
  Compression: How much salient information is conserved as the sequence length is compressed.
  Generation: The downstream latent diffusion model (LDM) that is trained to generate these representations. Higher latent dimension makes this process more difficult.

# Latent Action Model
Questions to consider:
  1. Enforce causality or block masking?
  2. Input tokenized latents or the raw waveform?
  3. Latent action cardinality?

Because I am compute poor I decided to train my Latent Action Model on the continuous tokenized latents from the previous step, instead of the raw waveform (which is what Genie did). I am likely leaving some performance on the floor by doing this, but for the sake of speed and progress, I think its a fair tradeoff. Plus, because of my somewhat narrow scope, the tokenizer is actually quite good, so hopefully the information being lost isn't crucial to this step. It also brings a substantial win for the sequence length. Training the model over 10s of data would be 160,000 samples using the waveform but only 80 - 500 for the different tokenizers.

Since I'm not going for real-time action controlled generation, causality isn't strictly necessary anywhere in the model. I plan on using this primarily for editing and inpainting, which corresponds better to random/segmented masking instead of causal masking. Causal masking has also been empirically shown to make generation more difficult. However some form of masking is required to ensure the latent actions provide important information to the model that it cannot get from peaking ahead.

# Dynamics Model


# All Together Now

# For my not so technical friends
