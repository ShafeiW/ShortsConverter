# YouTube Shorts Converter

This program automatically converts YouTube videos into Shorts format with generated captions and YouTube integration.

## Features

- Download YouTube videos
- Convert videos to Shorts format (9:16 aspect ratio)
- Generate automatic captions using Whisper AI
- Upload directly to YouTube
- Maintain video quality while fitting Shorts format
- Add stylish captions with background stroke for better readability

## Setup

1. Install Python 3.8 or higher
2. Install required packages:
   ```bash
   pip install -r requirements.txt
   ```

3. Set up YouTube API credentials:
   - Go to the [Google Cloud Console](https://console.cloud.google.com/)
   - Create a new project
   - Enable the YouTube Data API v3
   - Create OAuth 2.0 credentials
   - Download the credentials and save as `client_secrets.json` in the project directory

## Usage

Run the program with a YouTube URL:
```bash
python shorts_converter.py "https://www.youtube.com/watch?v=VIDEO_ID"
```

The program will:
1. Download the video
2. Generate captions
3. Convert to Shorts format
4. Ask if you want to upload to YouTube
5. Clean up temporary files

## Notes

- The first time you upload to YouTube, you'll need to authenticate through your browser
- Videos are uploaded as private by default
- Captions are generated using the Whisper AI model
- The program maintains the best possible quality while fitting the Shorts format

## Requirements

See `requirements.txt` for full list of dependencies.