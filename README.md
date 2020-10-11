# Vid Clip Generation

Generate a simple video. Default format is test_YYYY_MM_DD_uuid.mov running 4 seconds.


## Prerequisites

### Python > 3.7

### ffmpeg > 4.1
  - MacOS 
    - `brew install ffmpeg`
  - Debian Linux
    - `sudo apt install ffmpeg`
  - Windows
    - Install from https://www.ffmpeg.org/download.html and place exe in PATH

  The version of ffmpeg needs to be > v4.1 so that we can label the prores apropriately
  within moviepy. Add the following to your shell profile:
  `export FFMPEG_BINARY=$(which ffmpeg)`
  On Windows the path to ffmpeg must be added as an environment variable manually with the path

### ImageMagik
  - MacOS 
    - `brew install imagemagick`
  - Debian Linux
    - `sudo apt install imagemagick`
    - Note: for linux users the policy then needs to be updated to allow ImageMagick to write to certain locations.
    - `sudo cp ./policy.xml /etc/ImageMagick-6/policy.xml`
  - Windows
    - TODO

## Setup

- Clone this repo
- `cd ./vid-gen`
- `python -m venv venv_vidgen`
- `source venv_vidgen/bin/activate`
- `pip install -r requirements.txt`
- Done :) 

## Usage

`vid_clip_gen.py [-h] [--duration DURATION] [--codec CODEC] [--fr FR] [--name NAME] [--nonunique]`

```
optional arguments:
  -h, --help           show this help message and exit
  --duration DURATION  how many seconds the test video should run
  --codec CODEC        the video codec or wrapping scheme
  --fr FR              frame rate
  --name NAME          the first name of the file
  --nonunique          allow the same filename to be generated
```
