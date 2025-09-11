## Introduction
The inspiration for this was the original [Genie paper](https://arxiv.org/pdf/2402.15391) from Google DeepMind. If you haven't read it or seen [Genie3](https://deepmind.google/discover/blog/genie-3-a-new-frontier-for-world-models/) I highly recommend checking it out. The basic premis of these models is controllable generation without supervised labels, because collecting these labels (especially at fine granularity) can be quite expensive. What they do instead is train a Latent Action Model (LAM) to predict what action occurs between frames. Then they use these unsupervised labels for conditional generation. The main dataset used in the paper is a large collection of platformer videos collected from YouTube. They restrict the action space to 8 actions which they find produces actions for simple movement. A great idea that according to the paper both works and scales well!

## Problem Statement
I want to take the Genie approach and apply it to music. But instead of focusing on 'live' playability, to focus on semantic editing. I do this for two main reasons:
  1. There is a ton of backlash (understandably so) from artistic communnities on AI generated art. While I am participating in that from the AI side of things, I want my work to be interesting and acceptable to both sides.
  2. Editing (aka in-painting) should be a simpler task than pure generation (ignoring boundary conditions). And because I am quite data and compute limited, simpler tasks are preferable.

There are many high quality music generation models that let someone generate music from text, and if we go back far enough in time, even from sheet music. But at the moment text doesn't have sufficient fine grained control, and I'm not yet convinced its even the right medium for that level of control. Instead I propose to learn a set of actions for that very purpose, and most of the fun along the way will be discovering what those actions are. 

## Overview
Conceptually I follow the same approach as the initial Genie paper. I train an audio tokenizer, a latent action model, and finally a dynamics model. However each step in my training pipeline has some crucial deviations from Genie. I'll go through them one by one and explain what motivates them and the results they bring. I found an incredible dataset of old Jazz recordings. 

### Data


### Audio Tokenizer
From a conceptual POV I am anti quantization of continuous signals in tokenizers, why are we restricting the continuous nature of our data to some dumb integers in high dimensional space?? Why not use that continuous structure to our advantage?? He says yelling at no one in particular. An annoying looking man begins to say something about digital sensors being discrete by conception before he is pumbled by the audience, everyone applauds. So anyway I just don't like them, but they are the most performant tokenizers, so I understand why they are commonly used. At the onset of this project I spent some time fooling around with pretrained tokenziers and trying to build my own continuous tokenizers. It took a fair few failed attempts and bad results before I realized the main issue was that I was bad at googling and the solution already existed on ArXiv. Boom! [Diffusion Tokenizers](https://arxiv.org/pdf/2501.18593v1) why predict the entire latent space in one go when you can do it iteratively instead? A simple and profound idea that happened to be what I was messing with, just better and with many of the kinks worked out. I adjusted the architecture to work with 1D audio signals instead of 2D images, which mostly involved baking in the recent advances in audio tokenizers, and it pretty much worked out of the box! I love it when that happens. I highly recommend reading the Diffusion Tokenizers paper because it requires no specialized losses. Yes you heard me right, no collection of STFT losses, no GAN losses, no LIPSIS losses, or anything domain specific. Simply a diffusion objective that scales better than anyother image tokenizer. This worked quite well for low compression (16000hz -> 50 tokens). Below are some sample reconstructions.

| Original | Reconstruction |
|----------|----------------|
| <audio controls><source src="audio/0_real.wav" type="audio/wav"></audio> | <audio controls><source src="audio/0_recon.wav" type="audio/wav"></audio> |
| <audio controls><source src="audio/1_real.wav" type="audio/wav"></audio> | <audio controls><source src="audio/1_recon.wav" type="audio/wav"></audio> |
| <audio controls><source src="audio/2_real.wav" type="audio/wav"></audio> | <audio controls><source src="audio/2_recon.wav" type="audio/wav"></audio> |
| <audio controls><source src="audio/3_real.wav" type="audio/wav"></audio> | <audio controls><source src="audio/3_recon.wav" type="audio/wav"></audio> |
| <audio controls><source src="audio/5_real.wav" type="audio/wav"></audio> | <audio controls><source src="audio/5_recon.wav" type="audio/wav"></audio> |

Unfortunately naively extending this approach to higher levels of compression failed miserably. So did training a hierarchical tokenizer on top of the tokenized latents. [DC-AE](https://arxiv.org/pdf/2410.10733) to the rescue! The most insightful point in this paper is the realization that tokenizers have competing optimization targets that make learning difficult.
  1. Move information from the sequence to channels, which is not an easy task for convolution filters, and even for mlps/attention it has large gradients
  2. Compress the information, which typically has small gradients

### Latent Action Model
Questions to consider:
  1. Causal or no?
  2. Latents or raw waveform?
  3. How granular?
  4. How many actions?

Because I am compute poor I decided to train my Latent Action Model on the continuous tokenized latents from the previous step, instead of the raw waveform (which is what Genie did). I am likely leaving some performance on the floor by doing this, but for the sake of speed and progress, I think its a fair tradeoff. Plus, because of my somewhat narrow scope, the tokenizer is actually quite good, so hopefully the information being lost isn't crucial to this step.

Since I'm not going for real-time action controlled generation, causality isn't strictly necessary anywhere in the model. I plan on using this primarily for editing, which points to random/segmented masking instead of causal masking.

### Dynamics Model


### For my not so technical friends
