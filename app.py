import os
from typing import Tuple
import gradio as gr
import spaces
import spaces.config
import yt_dlp
import tempfile
import hashlib
import numpy as np
import soundfile as sf
import librosa
import matplotlib.pyplot as plt
from audio_separator.separator import Separator
from zero import dynGPU


separators = {
    "BS-RoFormer": Separator(output_dir=tempfile.gettempdir(), output_format="mp3"),
    "Mel-RoFormer": Separator(output_dir=tempfile.gettempdir(), output_format="mp3"),
    "HTDemucs-FT": Separator(output_dir=tempfile.gettempdir(), output_format="mp3"),
}


def load():
    separators["BS-RoFormer"].load_model("model_bs_roformer_ep_317_sdr_12.9755.ckpt")
    separators["Mel-RoFormer"].load_model(
        "model_mel_band_roformer_ep_3005_sdr_11.4360.ckpt"
    )
    separators["HTDemucs-FT"].load_model("htdemucs_ft.yaml")


# sometimes the network might be down, so we retry a few times
for _ in range(3):
    try:
        load()
        break
    except Exception as e:
        print(e)


def youtube(url: str) -> str:
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


def measure_duration(audio: str, model: str) -> int:
    y, sr = librosa.load(audio, sr=44100)
    return int(librosa.get_duration(y=y, sr=sr) / 3.0)


@dynGPU(duration=measure_duration)
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


def plot_spectrogram(audio: str):
    y, sr = librosa.load(audio, sr=44100)
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
    S_dB = librosa.power_to_db(S, ref=np.max)
    fig = plt.figure(figsize=(15, 5))
    librosa.display.specshow(S_dB, sr=sr, x_axis="time", y_axis="mel")
    plt.colorbar(format="%+2.0f dB")
    plt.title("Mel-frequency spectrogram")
    fig.tight_layout()
    return fig


with gr.Blocks() as app:
    with open(os.path.join(os.path.dirname(__file__), "README.md"), "r") as f:
        README = f.read()
        # remove yaml front matter
        blocks = README.split("---")
        if len(blocks) > 1:
            README = "---".join(blocks[2:])

    gr.Markdown(README)

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
            vocals = gr.Audio(
                label="Vocals", format="mp3", type="filepath", interactive=False
            )
        with gr.Column():
            bgm = gr.Audio(
                label="Background", format="mp3", type="filepath", interactive=False
            )

    with gr.Row():
        with gr.Column():
            vocal_spec = gr.Plot(label="Vocal spectrogram")
        with gr.Column():
            bgm_spec = gr.Plot(label="Background spectrogram")

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
        outputs=[vocals, bgm],
        api_name="separate",
    ).success(
        fn=plot_spectrogram,
        inputs=[vocals],
        outputs=[vocal_spec],
    ).success(
        fn=plot_spectrogram,
        inputs=[bgm],
        outputs=[bgm_spec],
    )

    yt_btn.click(
        fn=youtube,
        inputs=[yt],
        outputs=[audio],
    )

    app.launch(show_error=True)
