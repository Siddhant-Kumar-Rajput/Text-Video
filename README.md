# Text-Video

Generate simple 2D videos from long-form text prompts. The project ships with a minimal Flask web app that accepts a descriptive story or script, wraps it into readable segments, and produces an MP4 slideshow that highlights each line.

## Features

- 📄 Accepts paragraphs of text via a polished web UI
- 🎞️ Creates MP4 videos using Pillow and ImageIO without external rendering tools
- 📥 Provides in-browser playback and a download link for the generated video

## Getting started

1. Create and activate a virtual environment (optional but recommended).
2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Run the development server:

   ```bash
   flask --app app run --host 0.0.0.0 --port 5000 --debug
   ```

4. Visit [http://localhost:5000](http://localhost:5000) and enter a long-form description to generate a video.

Generated videos are stored in `static/videos/` and are automatically served as static assets.

## Notes

- The video generator uses a basic highlight animation that cycles through each wrapped line in your text. Adjust the settings in `generate_video_from_text` within `app.py` to experiment with other effects (frame rate, colors, timing, etc.).
- Videos are rendered using FFmpeg through `imageio`. A standard FFmpeg build is included with the `imageio` dependency via the `imageio-ffmpeg` plugin. If you encounter codec errors, run `python -c "import imageio; imageio.plugins.ffmpeg.download()"` to fetch an updated binary.
