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
I found an incredible dataset of Jazz recordings called [JazzSet](https://www.reddit.com/r/datasets/comments/1b73vz3/jazzset_large_audio_dataset_with_instrumentation/?utm_source=chatgpt.com) on Reddit. The recordings date from the late 19th centrury to the early 21st and vary substantially in quality. Data quality is curcial in training a strong model so I built a filtering pipeline to retain the highest quality recordings from the dataset. It's important to note that rejecting all "old" songs would have done a somewhat reasonable job of improving the overall audio quality, but at a great cost to data coverage. Instead I trained a simple model to help with filtering. I initially experimented with denoisers and similar models to clean up the audio quality instead of sacrificing data, but nothing I tried worked, and I didn't want to derail this project into training a denoiser before it even got started. There is plenty of derailing ahead.

## Data Filtering
I resampled all of the recordings to 16000Hz to reduce the computational training burden while maintaining reasonable peak audio quality. I randomly sampled ~200 songs, took the first 5 seconds from each, and labeled the quality as good or bad. The sampling was biased by age of the recording, anything somewhat recent had high quality so I didn't want to fill my training set with good quality recordings and bias the model accordingly. I trained an ExtraTrees model on simple statistics calculated over frequency bands from the Mel Spectrogram of the 5 second cuts. After training the model had ~80% accuracy which I deemed good enough for filtering. To filter, I calculated the same feature set over the entire dataset, predicted the probability of each 5 second segment being good, applied a gaussian filter to the probabilities, and passed a song through if the average probability was above 55%. I decided to pass/fail entire songs instead of their segments because I didn't want to deal with stitching segments together (despite this being common in NLP pre-training) and I didn't know what the final generative models context length would be (i.e. how much stitching would impact training). In hindsight maintaining sufficiently long (>30s) clips that were high quality likely would have increased my dataset size without too much overhead.

Here is an example of the difference in audio quality, the crackaling is a characateristic of old recording equipment and should be excluded from the training set.

| Good | Bad |
|------|-----|
| <audio controls><source src="samples/mel/JV-36144-1957-QmbSPzr8VX8LUrnatKVGRK9G9wZuVxah5VdMBgXVTpNBDn.wav-TS485813.wav" type="audio/wav"></audio> | <audio controls><source src="samples/mel/JV-12-1916-QmYZJNDBn5WPRNakbBi9UujN8dMrJo3n8vpaHo8RNon4xt.wav-JV-12-1916-QmYZJNDBn5WPRNakbBi9UujN8dMrJo3n8vpaHo8RNon4xt.wav" type="audio/wav"></audio> |

Here are the Mel Spectrograms for the first 5 seconds of each song that were used to generate the features for the quality classifier. The probability predicted by the classifier is in the title of each image. Notice how there is more color and blurred pixels in the second image, some of this comes from the crackle in the recording.

| <img src="samples/mel/good.png" width="95%"/> | <img src="samples/mel/bad.png" width="95%"/> |

# Audio Tokenizer
## Architectures
From a conceptual POV I am anti quantization of continuous signals in tokenizers, why are we restricting the continuous nature of our data to some dumb integers in high dimensional space?? Why not use that continuous structure to our advantage?? He yells a little too loudly at no one in particular. An annoying looking man begins to say something about digital sensors being discrete by design before he is pummeled by the audience, everyone applauds. So anyway, I just don't like them conceptually, but they are the most performant tokenizers, so I understand why they are so commonly used.

At the onset of this project I spent some time fooling around with pretrained tokenziers and trying to build my own continuous tokenizers. It took a fair few failed attempts and mediocore results before I realized the main issue was that I was bad at googling and the solution already existed on ArXiv. Boom! [Diffusion Tokenizers](https://arxiv.org/pdf/2501.18593v1) why predict the entire latent space in one go when you can do it iteratively instead? A simple and profound idea that happened to be what I was messing with, just better and with many of the kinks worked out. I adjusted the architecture to work with 1D audio signals instead of 2D images, which mostly involved baking in the recent advances in audio tokenizers, and it pretty much worked out of the box! I love it when that happens. I highly recommend reading the Diffusion Tokenizers paper because it requires no specialized losses. Yes you heard me right, no collection of STFT losses, no GAN losses, no LIPSIS losses, or anything domain specific. Simply a diffusion objective that scales better than anyother image tokenizer. This worked quite well for low compression (16000Hz -> 50 tokens). Below are some sample reconstructions.

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
  
The different gradient magnitudes are an explanation for why adding additional blocks to enocders does not simply give better compression. The gradient signal is dominated by moving information around rather than better compression. DC-AE proposes to solve this problem by handling Process 1 with [Pixel Shuffle](https://docs.pytorch.org/docs/stable/generated/torch.nn.PixelShuffle.html), a technique that has been around a few years for better upsampling --you may have used this technique when training u-nets for segmentation. There are no learned parameters in this process, so the large gradients disappear. Then the compression is handled by a similar but learned residual transformation with small gradients. With this new approach the model only needs to learn what information to add rather than how to move it around to the further compressed representation. Integrating DC-AE removed the training bottleneck greatly increasing reconstruction quality. Unfortunately it also degraded controlability, reconstruction samples were high quality but quite different from the original sample. The model was essentially generating high quality samples at random rather than being conditioned on the input signal. Thankfully a few simple architectual modifications greatly improved conditioning, particularly AdaLN-Zero and improved skip connections.

## AdaLN-Zero and Skip Connections
The consistency model used by DiTo implements long range U-Net skip connections for every layer of every block of computation. They are passed from encoder to decoder and used to preserve fine grained information, allowing the model to focus on compressing global information. One issue with this approach is that right after downsampling that signal can be noisy, the model has had little time to refine the representation, and yet this noisy representation is being passed over to the decoder. A simple fix is to only select the last layer prior to downsampling as a skip connection, as it is the most refined representation at its scale, and pass that to all layers in the decoder with the same scale. This also reduces memory usage as skip connections are only being stored for every downsampling operation, not for every layer.

[AdaLN-Zero](https://arxiv.org/pdf/2212.09748) is a great and efficient technique for conditional generation. The conditioning signal is projected into scale, shift, and gate parameters that are used to modulate the computation of a residual block.

Below are some sample reconstructions from the high level tokenizer (16000Hz -> ~8 tokens). Notice that the musical themes are consitent between the original and reconstruction but that the audio quality varies. This is because the tokenizer is too small and is still early in training.

| Original | Reconstruction |
|----------|----------------|
| <audio controls><source src="samples/high/2_real.wav" type="audio/wav"></audio> | <audio controls><source src="samples/high/2_recon.wav" type="audio/wav"></audio> |
| <audio controls><source src="samples/high/3_real.wav" type="audio/wav"></audio> | <audio controls><source src="samples/high/3_recon.wav" type="audio/wav"></audio> |
| <audio controls><source src="samples/high/7_real.wav" type="audio/wav"></audio> | <audio controls><source src="samples/high/7_recon.wav" type="audio/wav"></audio> |

To force the model to learn higher level and more longterm features, I doubled the inpute sequence length to 2 seconds and then later increased it to 5 seconds. This was faster and more computationally efficient for my setup than starting training with 5 second segments. Scaling up the model and training will improve audio quality which would likely improve downstream learning. However, I have limitied compute and time, and these reconstructions maintain the most salient information which is what is required for the next stage.

## Latent Shape
Taking advantage of the architectural modifications from DC-AE makes trading compression for latent dimension more tractable. The next question becomes what latent dimension has the best trade-off between compression and generation. Lets define what I mean by each of those terms in this specific context.
  Compression: How much salient information is conserved as the sequence length is compressed.
  Generation: How tractable is training a latent diffusion model (LDM) to generate these representations. Larger latent dimension makes this process more difficult.

The first tokenizer I trained compressed 16000Hz into 50 continuous tokens of dimension 128. Those with a background in image generation may back away in fear from a latent dimension of 128, most image tokenizers have between 4 and 16 dimensions, but 128 is standard in audio. Its easy to assume this is to compensate for audio tokenizers having larger compression ratios, but it is more likely due to the weakness of audio tokenizers. Image generation has made much more progress than audio and thus the tokenizers are better suited for the task. I improved the architecture and doubled the latent dimension as I scaled up the compression giving a model that compresses 16000Hz into ~8 tokens of dimension 256. Noticing that the reconstruction quality was substantially better than expected, I repeatedly retrained the model with the same spatial compression but reducing the latent dimension. First to 128 and then to 64 with no noticeable drop in quality. Crucially, this is all without scaling the decoder to compensate for the smaller latent representation. As shown in the DiTo paper, scaling the decoder would scale reconstruction quality and controlability performance.

# Latent Action Model
Questions to consider:
  1. Input tokenized latents or the raw waveform?
  2. Enforce causality or block masking?
  3. What do we want out action space to be?

## Input
Because I am compute poor I decided to train my Latent Action Model on the continuous tokenized latents from the previous step, instead of the raw waveform (which is what Genie did). I am likely leaving some performance on the floor by doing this, but for the sake of speed and progress, I think its a fair tradeoff. Plus, because of my somewhat narrow scope, the tokenizer is actually quite good, so hopefully the information being lost isn't crucial to this step. It also brings a substantial win for the sequence length. Training the model over 10s of data would be 160,000 samples using the waveform but only 80 - 500 for the different tokenizers.

## Masking
Since I'm not going for real-time action controlled generation, causality isn't strictly necessary anywhere in the model. I plan on using this primarily for editing and inpainting, which corresponds better to random/segmented masking instead of causal masking. Causal masking has also been empirically shown to make generation more difficult. However some form of masking is required to ensure the latent actions provide important information to the model that it cannot get from peaking ahead.

## Action Space
The latent action space cardinality must be defined prior to training. It can always be reduced for inference, but with the Genie setup it cannot be grown progressively during training. Okay thats slightly misleading, one could stitch on additional codebook slots and continue training, but theres no learning of the codebook size here. An interesting research direction could be learning the codebook size as some functional of information gain. But for my purpose, enforcing a maximum cardinality is actually crucial. The intuition being that a small number of actions should force the model to learn more meaningful and interpretable actions. One could imagine a small codebook learning actions like: guitar solo, melancholy tune, up-beat melody; while a large codebook could learn much more granular actions like: piano plays c-note, increase tempo, play hi-hat. Although both approaches are worthwile, I am much more interested in the first set of actions than the second. The first set opens the door to people with musical sense but less musical talent (think [Rick Ruben](https://www.youtube.com/watch?v=h5EV-JCqAZc). The second set could allow highly talent musicians to play a much wider set of instruments without having to learn them (although imagining building that UI, bleh). The keen-eyed readers may have noticed that only some of the examples included instruments. With no additional conditioning it is likely that the latent actions will pertain to the most informative actions, which certainly includes instruments. This is an important point worth thinking about because it has a huge impact on the final model. One approach is to provide the set of instruments being played during a segment as additional conditioning, that way the model learns actions on top of instruments. Then in inference if an action like solo is learned, the user can select what instrument (from the conditioning set) should play that solo.

# Dynamics Model


# All Together Now

# For my not so technical friends
