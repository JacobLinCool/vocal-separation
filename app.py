import os
from typing import Tuple
import gradio as gr
import spaces
import yt_dlp
import tempfile
import hashlib
from audio_separator.separator import Separator

MODEL = "model_bs_roformer_ep_317_sdr_12.9755.ckpt"

separator = Separator()
separator.load_model(MODEL)


def use_yt_url(url: str) -> str:
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


@spaces.GPU(duration=120)
def separate(audio: str) -> Tuple[str, str]:
    outs = separator.separate(audio)
    return outs[1], outs[0]


with gr.Blocks() as app:

    gr.Markdown(
        f"""
        # BS-RoFormer Vocal Separation

        This is a demo of the BS-RoFormer vocal separation model, which is the SOTA model for vocal separation. ([SDX23](https://arxiv.org/abs/2309.02612))
        
        Upload an audio file and the model will separate the vocals from the background music.

        > The model (`{MODEL}`) is trained by the [UVR project](https://github.com/Anjok07/ultimatevocalremovergui).
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
            yt_btn = gr.Button("Use this youtube URL")

    btn = gr.Button("Separate", variant="primary")

    with gr.Row():
        with gr.Column():
            vovals = gr.Audio(label="Vocals", format="mp3")
        with gr.Column():
            bgm = gr.Audio(label="Background", format="mp3")

    btn.click(
        fn=separate,
        inputs=[audio],
        outputs=[vovals, bgm],
    )

    yt_btn.click(
        fn=use_yt_url,
        inputs=[yt],
        outputs=[audio],
    )

    app.launch()
