import base64
from pathlib import Path

from IPython import display as ipythondisplay


def show_videos(video_path: str = "", prefix: str = "") -> None:
    """
    Taken from https://github.com/eleurent/highway-env

    :param video_path: Path to the folder containing videos
    :param prefix: Filter the video, showing only the only starting with this prefix
    """
    html = []
    for mp4 in Path(video_path).glob(f"{prefix}*.mp4"):
        video_b64 = base64.b64encode(mp4.read_bytes())
        html.append(
            """<video alt="{}" autoplay
                    loop controls style="height: 400px;">
                    <source src="data:video/mp4;base64,{}" type="video/mp4" />
                </video>""".format(
                mp4, video_b64.decode("ascii")
            )
        )
    ipythondisplay.display(ipythondisplay.HTML(data="<br>".join(html)))
