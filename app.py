import os
from typing import Tuple
import gradio as gr
import spaces
import yt_dlp
import tempfile
import hashlib
import numpy as np
import soundfile as sf
from audio_separator.separator import Separator


separators = {
    "BS-RoFormer": Separator(output_dir=tempfile.gettempdir(), output_format="mp3"),
    "Mel-RoFormer": Separator(output_dir=tempfile.gettempdir(), output_format="mp3"),
    "HTDemucs-FT": Separator(output_dir=tempfile.gettempdir(), output_format="mp3"),
}

separators["BS-RoFormer"].load_model("model_bs_roformer_ep_317_sdr_12.9755.ckpt")
separators["Mel-RoFormer"].load_model(
    "model_mel_band_roformer_ep_3005_sdr_11.4360.ckpt"
)
separators["HTDemucs-FT"].load_model("htdemucs_ft.yaml")


def use_yt_url(url: str) -> str:
    if not url:
        raise gr.Error("Please input a YouTube URL")

    hash = hashlib.md5(url.encode()).hexdigest()
    tmp_file = os.path.join(tempfile.gettempdir(), f"{hash}")

    ydl_opts = {
        "format": "bestaudio/best",
        "outtmpl": tmp_file,
        "postprocessors": [
            {
                "key": "FFmpegExtractAudio",
                "preferredcodec": "mp3",
                "preferredquality": "192",
            }
        ],
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])

    return tmp_file + ".mp3"


def merge(outs):
    bgm = np.sum(np.array([sf.read(out)[0] for out in outs]), axis=0)
    tmp_file = os.path.join(tempfile.gettempdir(), f"{outs[0].split('/')[-1]}_merged")
    sf.write(tmp_file + ".mp3", bgm, 44100)
    return tmp_file + ".mp3"


@spaces.GPU(duration=120)
def separate(audio: str, model: str) -> Tuple[str, str]:
    separator = separators[model]
    outs = separator.separate(audio)
    outs = [os.path.join(tempfile.gettempdir(), out) for out in outs]
    # roformers
    if len(outs) == 2:
        return outs[1], outs[0]
    # demucs
    if len(outs) == 4:
        bgm = merge(outs[:3])
        return outs[3], bgm
    raise gr.Error("Unknown output format")


with gr.Blocks() as app:

    gr.Markdown(
        f"""
        # BS-RoFormer Vocal Separation

        This is a demo of the BS-RoFormer vocal separation model, which is the SOTA model for vocal separation. ([MDX23](https://www.aicrowd.com/challenges/sound-demixing-challenge-2023/problems/music-demixing-track-mdx-23/leaderboards))

        Upload an audio file and the model will separate the vocals from the background music.

        For comparison, you can also try the Mel-RoFormer model (a variant of BS-RoFormer) and the popular HTDemucs FT model.

        > The models are trained by the [UVR project](https://github.com/Anjok07/ultimatevocalremovergui).

        > The code of this app is available on [GitHub](https://github.com/JacobLinCool/BS-RoFormer-app), any contributions should go there. Hugging Face Space is force pushed by GitHub Actions.
        """
    )

    with gr.Row():
        with gr.Column():
            gr.Markdown("## Upload an audio file")
            audio = gr.Audio(label="Upload an audio file", type="filepath")
        with gr.Column():
            gr.Markdown(
                "## or use a YouTube URL\n\nTry something on [The First Take](https://www.youtube.com/@The_FirstTake)?"
            )
            yt = gr.Textbox(
                label="YouTube URL", placeholder="https://www.youtube.com/watch?v=..."
            )
            yt_btn = gr.Button("Use this YouTube URL")

    with gr.Row():
        model = gr.Radio(
            label="Select a model",
            choices=[s for s in separators.keys()],
            value="BS-RoFormer",
        )
        btn = gr.Button("Separate", variant="primary")

    with gr.Row():
        with gr.Column():
            vovals = gr.Audio(label="Vocals", format="mp3")
        with gr.Column():
            bgm = gr.Audio(label="Background", format="mp3")

    gr.Examples(
        examples=[
            # I don't have any good examples, please contribute some!
            # Suno's generated musix seems to have too many artifacts
        ],
        inputs=[audio],
    )

    gr.Markdown(
        """
        - BS-RoFormer: https://arxiv.org/abs/2309.02612
        - Mel-RoFormer: https://arxiv.org/abs/2310.01809
        """
    )

    btn.click(
        fn=separate,
        inputs=[audio, model],
        outputs=[vovals, bgm],
    )

    yt_btn.click(
        fn=use_yt_url,
        inputs=[yt],
        outputs=[audio],
    )

    app.launch()
