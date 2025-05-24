import os
import sys
from typing import Optional, List, Tuple
import yt_dlp
from moviepy.editor import VideoFileClip, TextClip, CompositeVideoClip, ImageClip, ColorClip
from faster_whisper import WhisperModel
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload
import pickle
from dotenv import load_dotenv
import re
import ssl
import certifi
import glob
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import tempfile
import cv2
from scipy.signal import find_peaks

# Load environment variables
load_dotenv()

class ShortsConverter:
    def __init__(self):
        self.youtube = None
        self.credentials = None
        self.SCOPES = ['https://www.googleapis.com/auth/youtube.upload']
        
    def sanitize_filename(self, filename: str) -> str:
        """Sanitize filename to remove invalid characters."""
        # Remove invalid characters
        filename = re.sub(r'[<>:"/\\|?*？]', '', filename)
        # Replace spaces with underscores
        filename = filename.replace(' ', '_')
        # Remove any non-ASCII characters
        filename = ''.join(char for char in filename if ord(char) < 128)
        return filename
        
    def find_downloaded_file(self, base_name: str) -> str:
        """Find the downloaded file even if it has special characters."""
        # Get all mp4 files in the current directory
        mp4_files = glob.glob("*.mp4")
        # Find the most recently modified file
        if mp4_files:
            return max(mp4_files, key=os.path.getmtime)
        return None
        
    def download_video(self, url: str) -> str:
        """Download video from YouTube URL."""
        try:
            ydl_opts = {
                'format': 'best[ext=mp4]',
                'outtmpl': '%(title)s.%(ext)s',
                'quiet': False,
                'no_warnings': False,
                'extract_flat': False,
                'http_headers': {
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36',
                    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
                    'Accept-Language': 'en-US,en;q=0.5',
                },
                'nocheckcertificate': True,
                'ignoreerrors': False,
                'no_color': False,
                'geo_bypass': True,
                'geo_verification_proxy': None,
                'socket_timeout': 30,
                'retries': 10,
                'fragment_retries': 10,
                'extractor_retries': 10,
                'file_access_retries': 10,
                'extractor_args': {
                    'youtube': {
                        'skip': ['dash', 'hls'],
                        'player_client': ['android'],
                        'player_skip': ['webpage', 'configs'],
                    }
                },
                'source_address': '0.0.0.0',
                'force_generic_extractor': False,
                'allow_unplayable_formats': False,
                'prefer_insecure': True,
                'legacy_server_connect': True
            }
            
            # Create a custom SSL context that doesn't verify certificates
            ssl._create_default_https_context = ssl._create_unverified_context
            
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                print("Fetching video information...")
                try:
                    # First try to get video info without downloading
                    info = ydl.extract_info(url, download=False)
                    if not info:
                        raise Exception("Could not extract video information")
                        
                    print(f"Downloading: {info['title']}")
                    
                    # Now download the video
                    ydl.download([url])
                    
                    # Get the output filename and sanitize it
                    output_path = f"{info['title']}.mp4"
                    sanitized_path = self.sanitize_filename(output_path)
                    
                    # Try to find the downloaded file
                    downloaded_file = self.find_downloaded_file(info['title'])
                    if downloaded_file:
                        print(f"Found downloaded file: {downloaded_file}")
                        # Rename the file to the sanitized name
                        os.rename(downloaded_file, sanitized_path)
                        print(f"Renamed to: {sanitized_path}")
                        return sanitized_path
                    else:
                        raise Exception("Could not find downloaded file")
                        
                except Exception as e:
                    print(f"Error during download: {str(e)}")
                    raise
                
        except Exception as e:
            print(f"Error downloading video: {str(e)}")
            print("\nTroubleshooting steps:")
            print("1. Try using a different video URL")
            print("2. Check if the video is age-restricted")
            print("3. Make sure you have a stable internet connection")
            print("4. Try updating yt-dlp: pip install -U yt-dlp")
            print("5. Check if the video is available in your region")
            print("6. Try running: pip install --upgrade certifi")
            return None

    def generate_captions(self, video_path: str) -> list:
        """Generate captions using Whisper model."""
        model = WhisperModel("base", device="cpu", compute_type="int8")
        segments, _ = model.transcribe(video_path)
        return [(segment.start, segment.end, segment.text) for segment in segments]

    def resize_frame(self, frame, target_size):
        """Resize a frame using PIL's LANCZOS resampling."""
        img = Image.fromarray(frame)
        resized = img.resize(target_size, Image.Resampling.LANCZOS)
        return np.array(resized)

    def create_caption_image(self, text, size, fontsize=70):
        """Create a caption image using PIL instead of ImageMagick."""
        # Create a transparent image
        img = Image.new('RGBA', size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(img)
        
        # Try to load a system font
        try:
            font = ImageFont.truetype("arial.ttf", fontsize)
        except:
            font = ImageFont.load_default()
        
        # Calculate text size and position
        text_bbox = draw.textbbox((0, 0), text, font=font)
        text_width = text_bbox[2] - text_bbox[0]
        text_height = text_bbox[3] - text_bbox[1]
        
        x = (size[0] - text_width) // 2
        y = int(size[1] * 0.66 - text_height // 2)  # 2/3 down the screen
        
        # Draw text with stroke
        for offset_x, offset_y in [(-2, -2), (-2, 2), (2, -2), (2, 2)]:
            draw.text((x + offset_x, y + offset_y), text, font=font, fill='black')
        draw.text((x, y), text, font=font, fill='white')
        
        return np.array(img)

    def analyze_video_segments(self, video_path: str, max_duration: float = 60.0) -> List[Tuple[float, float]]:
        """Analyze video to find interesting segments for shorts."""
        video = VideoFileClip(video_path)
        fps = video.fps
        
        # Sample frames at 1-second intervals instead of every frame
        sample_interval = int(fps)
        total_frames = int(video.duration)
        
        # Calculate motion scores for sampled frames
        motion_scores = []
        prev_frame = None
        
        print("Analyzing video content...")
        for t in range(0, total_frames, sample_interval):
            if t % 10 == 0:  # Progress indicator
                print(f"Progress: {t}/{total_frames} seconds")
                
            frame = video.get_frame(t/fps)
            # Convert to grayscale for faster processing
            frame_gray = cv2.cvtColor(frame.astype(np.uint8), cv2.COLOR_RGB2GRAY)
            
            if prev_frame is not None:
                # Calculate motion between frames using grayscale
                diff = cv2.absdiff(frame_gray, prev_frame)
                motion_score = np.mean(diff)
                motion_scores.append(motion_score)
            prev_frame = frame_gray
        
        motion_scores = np.array(motion_scores)
        
        # Find peaks in motion scores with minimum distance
        min_distance = int(5)  # Minimum 5 seconds between peaks
        peaks, _ = find_peaks(motion_scores, distance=min_distance)
        
        # Convert peaks to timestamps
        timestamps = peaks * sample_interval / fps
        
        # Create segments around peaks
        segments = []
        for peak in timestamps:
            start_time = max(0, peak - max_duration/2)
            end_time = min(video.duration, start_time + max_duration)
            
            # Adjust if segment is too short
            if end_time - start_time < max_duration:
                start_time = max(0, end_time - max_duration)
            
            segments.append((start_time, end_time))
            
            # Break if we have enough segments
            if len(segments) >= 3:
                break
        
        video.close()
        return segments

    def create_shorts_format(self, video_path: str, captions: list, start_time: float, end_time: float) -> str:
        """Convert video segment to Shorts format with captions."""
        print(f"Processing segment from {start_time:.1f}s to {end_time:.1f}s...")
        video = VideoFileClip(video_path).subclip(start_time, end_time)
        
        # Resize to 9:16 aspect ratio (1080x1920)
        target_width = 1080
        target_height = 1920
        
        # Calculate new dimensions while maintaining aspect ratio
        aspect_ratio = video.size[0] / video.size[1]
        if aspect_ratio > 9/16:
            # Video is wider than 9:16
            new_width = target_width
            new_height = int(new_width / aspect_ratio)
            # Scale up to fill height
            scale = target_height / new_height
            new_width = int(new_width * scale)
            new_height = target_height
        else:
            # Video is taller than 9:16
            new_height = target_height
            new_width = int(new_height * aspect_ratio)
            # Scale up to fill width
            scale = target_width / new_width
            new_width = target_width
            new_height = int(new_height * scale)
            
        # Resize main video
        def resize_main_video(frame):
            return self.resize_frame(frame, (new_width, new_height))
        resized_video = video.fl_image(resize_main_video)
        
        # Create black background
        background = ColorClip(size=(target_width, target_height), color=(0, 0, 0))
        background = background.set_duration(video.duration)
        
        # Center the video on the background
        x_offset = (target_width - new_width) // 2
        y_offset = (target_height - new_height) // 2
        resized_video = resized_video.set_position((x_offset, y_offset))
        
        # Filter captions for this segment
        segment_captions = [(start, end, text) for start, end, text in captions 
                          if start >= start_time and end <= end_time]
        
        # Add captions using PIL
        caption_clips = []
        for start, end, text in segment_captions:
            # Adjust caption timing to segment
            adjusted_start = start - start_time
            adjusted_end = end - start_time
            
            # Create caption image
            caption_img = self.create_caption_image(text, (target_width, target_height))
            
            # Create a clip from the image array
            caption_clip = ImageClip(caption_img)
            caption_clip = caption_clip.set_duration(adjusted_end - adjusted_start)
            caption_clip = caption_clip.set_start(adjusted_start)
            caption_clip = caption_clip.set_position(('center', 'bottom'))
            caption_clips.append(caption_clip)
        
        # Combine video and captions
        final = CompositeVideoClip([background, resized_video] + caption_clips)
        
        # Save the final video
        output_path = f"shorts_output_{int(start_time)}.mp4"
        print(f"Rendering video to {output_path}...")
        final.write_videofile(output_path, codec='libx264', audio_codec='aac', 
                            threads=4,  # Use multiple threads
                            preset='ultrafast')  # Use faster encoding preset
        
        # Clean up
        video.close()
        final.close()
        for clip in caption_clips:
            clip.close()
        
        return output_path

    def authenticate_youtube(self):
        """Authenticate with YouTube API."""
        creds = None
        if os.path.exists('token.pickle'):
            with open('token.pickle', 'rb') as token:
                creds = pickle.load(token)
                
        if not creds or not creds.valid:
            if creds and creds.expired and creds.refresh_token:
                creds.refresh(Request())
            else:
                flow = InstalledAppFlow.from_client_secrets_file(
                    'client_secrets.json', self.SCOPES)
                creds = flow.run_local_server(port=8080)
            with open('token.pickle', 'wb') as token:
                pickle.dump(creds, token)
                
        self.youtube = build('youtube', 'v3', credentials=creds)

    def upload_to_youtube(self, video_path: str, title: str, description: str):
        """Upload video to YouTube."""
        if not self.youtube:
            self.authenticate_youtube()
            
        request_body = {
            'snippet': {
                'title': title,
                'description': description,
                'tags': ['shorts', 'youtube shorts'],
                'categoryId': '22'
            },
            'status': {
                'privacyStatus': 'private',
                'selfDeclaredMadeForKids': False
            }
        }
        
        mediaFile = MediaFileUpload(video_path, 
                                  mimetype='video/mp4',
                                  resumable=True)
        
        request = self.youtube.videos().insert(
            part=','.join(request_body.keys()),
            body=request_body,
            media_body=mediaFile
        )
        
        response = request.execute()
        return response

def main():
    if len(sys.argv) < 2:
        print("Usage: python shorts_converter.py <youtube_url>")
        return
        
    url = sys.argv[1]
    converter = ShortsConverter()
    
    # Download video
    print("Downloading video...")
    video_path = converter.download_video(url)
    if not video_path:
        print("Failed to download video")
        return
        
    # Generate captions
    print("Generating captions...")
    captions = converter.generate_captions(video_path)
    
    # Analyze video and find interesting segments
    print("Analyzing video for interesting segments...")
    segments = converter.analyze_video_segments(video_path)
    
    # Create Shorts for each segment
    print("Creating Shorts...")
    shorts_paths = []
    for i, (start_time, end_time) in enumerate(segments, 1):
        print(f"Creating Short {i}...")
        shorts_path = converter.create_shorts_format(video_path, captions, start_time, end_time)
        shorts_paths.append(shorts_path)
    
    # Upload to YouTube (optional)
    upload = input("Would you like to upload to YouTube? (y/n): ")
    if upload.lower() == 'y':
        for i, shorts_path in enumerate(shorts_paths, 1):
            title = input(f"Enter title for Short {i}: ")
            description = input(f"Enter description for Short {i}: ")
            print(f"Uploading Short {i} to YouTube...")
            response = converter.upload_to_youtube(shorts_path, title, description)
            print(f"Short {i} uploaded successfully! Video ID: {response['id']}")
    
    # Clean up
    os.remove(video_path)
    for path in shorts_paths:
        os.remove(path)
    print("Process completed successfully!")

if __name__ == "__main__":
    main() 