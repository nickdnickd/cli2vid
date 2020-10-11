""" Small test video creation module. 
    The purpose of this video is to generate small, uniquely named
    mp4's or op1a's that contain a simple pattern and random color"""

import gizeh
import moviepy.editor as mpy

from pymediainfo import MediaInfo

from datetime import datetime
import getpass

import argparse
import uuid
import math

from typing import Tuple, Optional, List

import ffmpeg
import os

import numpy as np

W = 24

DEFAULT_DURATION_S = 4
DEFAULT_VCODEC = "prores_ks"
DEFAULT_ACODEC = "pcm_s24le"
DEFAULT_FRAMERATE = 23.98
DEFAULT_FILENAMEROOT = "test"
DEFAULT_HD_WIDTH = 1920
DEFAULT_HD_HEIGHT = 1080
DEFAULT_AUDIO_RATE = 48000  # Samples / second


def make_frame(
    t,
    width,
    height,
    total_time_s,
    fill_color: Tuple[float, float, float],
    extra_fills: Optional[List[Tuple[float, float, float]]] = None,
):
    """Generate a frame given t of a circle that changes shape.
    http://zulko.github.io/moviepy/getting_started/videoclips.html"""

    surface = gizeh.Surface(width, height)  # width, height
    radius = (width / 2) * abs(math.sin(2 * math.pi * t / float(total_time_s)))
    circle = gizeh.circle(radius, xy=(int(width / 2), int(height / 2)), fill=fill_color)
    circle.draw(surface)

    if extra_fills:
        for idx, extra_fill in enumerate(extra_fills):
            # use the index to shift the x-y position and change the starting point of the radius
            pos = 1 + ((idx + 1) * 0.10)  # 10 percent in this direction
            direction_x = -1 if idx % 2 == 0 else 1
            direction_y = -1 if idx % 3 == 0 else 1
            this_xy = (
                int((width / 2) + (direction_x * pos * width / 2)),
                int((height / 2) + (direction_y * pos * height / 2)),
            )

            radius = (width / 2) * abs(
                math.sin(2 * math.pi * (t * pos) / float(total_time_s))
            )
            circle = gizeh.circle(radius, xy=this_xy, fill=extra_fill)
            circle.draw(surface)

    npimage = surface.get_npimage()

    # add light noise
    noise = np.random.randint(0, 2, size=(height, width, 3))

    return npimage + noise  # returns a 8-bit RGB array


def create_color_fill_from_uuid(uuid: uuid.UUID) -> Tuple[float, float, float]:
    r_ratio = float(uuid.bytes[0] >> 4) / 16.0
    g_ratio = float(uuid.bytes[1] >> 4) / 16.0
    b_ratio = float(uuid.bytes[2] >> 4) / 16.0
    return (r_ratio, g_ratio, b_ratio)


def generate_audio_track_files(
    mini_hash="",
    rate=DEFAULT_AUDIO_RATE,
    f=440.0,
    T=DEFAULT_DURATION_S,
    track_layout: Optional[Tuple[str, ...]] = None,
):
    """Generates temporary wav files and returns their locations."""
    import numpy as np
    import wavio

    t = np.linspace(0, int(T), int(T * rate), endpoint=False)

    # Define the intervals
    x = np.sin(2 * np.pi * f * t)
    x_3rd = np.sin(2 * np.pi * (f * 5.0 / 4.0) * t)
    x_5th = np.sin(2 * np.pi * (f * 3.0 / 2.0) * t)
    x_oct = np.sin(2 * np.pi * (f * 2.0) * t)
    x_bass = np.sin(2 * np.pi * (f / 2.0) * t)
    x_bass_pfifth = np.sin(2 * np.pi * (f * 3.0 / 4.0) * t)
    x_sil = np.zeros(len(t))

    # Define the tracks as wav files
    # This is the contribution standard but below we are adhereing
    # to what is already in baton

    # Contribution standard default layout
    if track_layout:
        tracks = track_layout
    else:
        return [], []  #  tracks = ("maj3", "sil", "maj5", "maj3", "bass", "oct")

    written_files = []

    for track_num, track in enumerate(tracks):
        track_file = f"{track}_a{track_num}_{mini_hash}.wav"  # TODO add prefix

        if track == "sil":
            track_data = np.vstack((x_sil, x_sil)).T
        elif track == "maj3":
            track_data = np.vstack((x, x_3rd)).T
        elif track == "maj5":
            track_data = np.vstack((x, x_5th)).T
        elif track == "oct":
            track_data = np.vstack((x, x_oct)).T
        elif track == "bass":
            track_data = np.vstack((x_bass_pfifth, x_bass)).T
        elif track == "mono":
            track_data = x

        wavio.write(track_file, track_data, rate, sampwidth=3)

        written_files.append(track_file)

    return tuple(written_files), tracks


def create_audio_streams(writen_files, track_layout, volume_reduction_db=23):
    """Takes a tuple of audio files that have been generated and
    converts them into a tuple of ffmpeg streams.
    A mono track needs less volume reduction"""

    return tuple(
        [
            ffmpeg.input(written_file).filter("volume", f"-{volume_reduction_db-3}dB")
            if audio_track == "mono"
            else ffmpeg.input(written_file).filter(
                "volume", f"-{volume_reduction_db}dB"
            )
            for written_file, audio_track in zip(writen_files, track_layout)
        ]
    )


def get_precise_video_duration(video_track_file):

    media_info = MediaInfo.parse(video_track_file)

    main_track = media_info.tracks[0]

    return main_track.duration


def generate_video(
    total_duration_s: int = DEFAULT_DURATION_S,
    vcodec: str = DEFAULT_VCODEC,
    acodec: str = DEFAULT_ACODEC,
    framerate: float = DEFAULT_FRAMERATE,
    filename_root: str = DEFAULT_FILENAMEROOT,
    frame_width: int = DEFAULT_HD_WIDTH,
    frame_height: int = DEFAULT_HD_HEIGHT,
    nonunique_filename: bool = False,
    audio_track_layout: Optional[Tuple[str, ...]] = None,
    use_faststart: bool = False,
    use_hapq: bool = False,
    num_extra_circles: int = 0,
    quicktime_support: bool = False,
):

    # Generate the unique information
    username = getpass.getuser()
    now = datetime.now()
    rand_uuid = uuid.uuid4()
    mini_hash = str(rand_uuid)[:5]

    fill = create_color_fill_from_uuid(rand_uuid)

    extra_fills = [
        create_color_fill_from_uuid(uuid.uuid4()) for _ in range(num_extra_circles)
    ]
    print(extra_fills)

    hoy = datetime.today()
    # five day uuid identifier
    video_track_file = (
        f"{filename_root}.mp4"
        if nonunique_filename
        else f"{filename_root}_{hoy.year}_{hoy.month}_{hoy.day}_v0_{mini_hash}.mp4"
    )

    output_filename = (
        f"{filename_root}.mp4"
        if nonunique_filename
        else f"{filename_root}_{hoy.year}_{hoy.month}_{hoy.day}_{mini_hash}.mp4"
    )

    # Be wary when converting text clips
    # https://stackoverflow.com/questions/42928765/convertnot-authorized-aaaa-error-constitute-c-readimage-453
    txt_clip = mpy.TextClip(
        f"Test Video Created by {username}\n{now}",
        fontsize=int(frame_width / 22),
        color="white",
    )

    txt_clip.set_pos("center").set_duration(total_duration_s)

    make_frame_fill = lambda t: make_frame(
        t, frame_width, frame_height, total_duration_s, fill, extra_fills
    )

    clip = mpy.VideoClip(make_frame_fill, duration=total_duration_s)

    video = mpy.CompositeVideoClip([clip, txt_clip]).set_duration(total_duration_s)

    # These are prores specific ffmpeg params
    if "prores" in vcodec:
        ffmpeg_params = [
            "-pix_fmt",
            "yuv422p10le",
            "-vtag",
            "apch",
            "-bsf:v",
            "prores_metadata=color_primaries=bt709:color_trc=bt709:colorspace=bt709",
            "-color_primaries",
            "bt709",
            "-color_trc",
            "bt709",
            "-colorspace",
            "bt709",
        ]
    elif use_hapq:
        ffmpeg_params = ["-format", "hap_q"]
    else:
        ffmpeg_params = []

    if quicktime_support:

        if not ffmpeg_params:
            ffmpeg_params = []

        ffmpeg_params.extend(["-pix_fmt", "yuv420p"])

    video.write_videofile(
        video_track_file, codec=vcodec, fps=framerate, ffmpeg_params=ffmpeg_params
    )

    # Note Moviepy cannot yet combine 24 bit audio clips
    # We must use wavio to write the audio track and use moviepy to write the video track

    # Furthermore, we need to get the exact duration of the video_track_file since
    # frames per second had a coarse ~20ms resolution whereas audio contains far
    # more sample resolution

    precise_video_duration_ms = get_precise_video_duration(video_track_file)

    precise_video_duration_s = precise_video_duration_ms / 1000.0
    audio_track_files, audio_layout = generate_audio_track_files(
        mini_hash=mini_hash, T=precise_video_duration_s, track_layout=audio_track_layout
    )

    ffmpeg_kwargs = {
        "timecode": "00:59:20:00",
        "vcodec": "copy",
        "acodec": acodec,
    }

    if use_faststart:
        ffmpeg_kwargs["movflags"] = "faststart"

    # ffmpeg -i test_2019_12_20_b8933.mov -i sine24_maj_t1.wav -i sine24_maj_t2.wav -map 0:0 -map 1:0 -map 2:0 -c copy  output.mov
    (
        ffmpeg.output(
            ffmpeg.input(video_track_file),
            *create_audio_streams(
                audio_track_files, audio_layout, volume_reduction_db=23
            ),
            output_filename,
            **ffmpeg_kwargs,
        ).run()
    )

    # Remove artifact files
    os.remove(video_track_file)
    for audio_track_file in audio_track_files:
        os.remove(audio_track_file)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Generate a simple video. Default format is "
        "test_YYYY_MM_DD_<uuid>.mov running 4 seconds."
    )
    parser.add_argument(
        "--duration",
        default=DEFAULT_DURATION_S,
        type=int,
        help="how many seconds the test video should run",
    )
    parser.add_argument("--vcodec", default=DEFAULT_VCODEC, help="the video codec")
    parser.add_argument("--acodec", default=DEFAULT_ACODEC, help="the audio codec")
    parser.add_argument(
        "--fr", type=float, default=DEFAULT_FRAMERATE, help="frame rate"
    )
    parser.add_argument(
        "--name", default=DEFAULT_FILENAMEROOT, help="the first name of the file"
    )
    parser.add_argument(
        "--nonunique",
        action="store_true",
        help="allow the same filename to be generated",
    )

    args = parser.parse_args()

    generate_video(
        total_duration_s=args.duration,
        vcodec=args.vcodec,
        acodec=args.acodec,
        framerate=args.fr,
        filename_root=args.name,
        nonunique_filename=args.nonunique,
    )
