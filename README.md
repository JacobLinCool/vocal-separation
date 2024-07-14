# Vocal Separation SOTA

[HuggingFace Spaces](https://huggingface.co/spaces/JacobLinCool/vocal-separation)

This is a demo for SOTA vocal separation models. Upload an audio file and the model will separate the vocals from the background music.

Based on the result of [MDX23](https://www.aicrowd.com/challenges/sound-demixing-challenge-2023/problems/music-demixing-track-mdx-23/leaderboards), the current SOTA model is [BS-RoFormer](https://arxiv.org/abs/2309.02612).

For comparison, you can also try the Mel-RoFormer model (a variant of BS-RoFormer) and the popular HTDemucs FT model.

## Models

- BS-RoFormer
- Mel-RoFormer
- HTDemucs FT

> The models are trained by the [UVR project](https://github.com/Anjok07/ultimatevocalremovergui).

> The code of this app is available on [GitHub](https://github.com/JacobLinCool/vocal-separation), any contributions should go there. Hugging Face Space is force pushed by GitHub Actions.
