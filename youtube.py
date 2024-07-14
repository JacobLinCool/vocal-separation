import os
import gradio as gr
from gradio_client import Client
import yt_dlp
import tempfile
import hashlib
import shutil


def youtube(url: str) -> str:
    if not url:
        raise gr.Error("Please input a YouTube URL")

    hash = hashlib.md5(url.encode()).hexdigest()
    tmp_file = os.path.join(tempfile.gettempdir(), f"{hash}")

    try:
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
    except Exception as e:
        print(e)
        try:
            ytdl = Client("JacobLinCool/yt-dlp")
            file = ytdl.predict(api_name="/download", url=url)
            shutil.move(file, tmp_file + ".mp3")
        except Exception as e:
            print(e)
            raise gr.Error(f"Failed to download YouTube audio from {url}")

    return tmp_file + ".mp3"
