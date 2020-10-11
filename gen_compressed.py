"""This is a config file to genere a file that should be acceptable
   by the Content Acquisitions """
import gen_video as vcg

# This is from the earth to the moon's format
# We need to specially handle mono tracks because they are quieter
audio_layout = ("maj3", "oct", "maj5", "mono", "mono", "mono", "mono", "mono", "mono")


vcg.generate_video(
    total_duration_s=15,
    vcodec="h264",
    acodec="mp3",
    frame_width=1280,
    frame_height=720,
    audio_track_layout=audio_layout,
    use_faststart=True,
    num_extra_circles=6,
)
