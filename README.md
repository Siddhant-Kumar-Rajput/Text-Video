# Text-Video

Generate atmospheric 2D videos from long-form text prompts. The project ships with a Flask web app that accepts a descriptive story or script, extracts key ideas, and produces an MP4 montage composed of animated gradients, contextual word clouds, and caption cards.

## Features

- 📄 Accepts paragraphs of text via a polished web UI with a built-in loading indicator while rendering
- 🖼️ Generates storyboards that blend gradients, word clouds, and highlighted captions for each scene
- 🎞️ Creates MP4 videos using Pillow, WordCloud, and ImageIO without external rendering tools
- 📥 Provides in-browser playback plus a prominent **Download file** button for the generated video

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

Once you submit your text, the UI displays a spinner and message until the MP4 is ready. When rendering finishes, the player appears alongside a download button labelled **Download file**.

## Notes

- The video generator now assembles each sentence into a dedicated "scene" by combining gradient backgrounds, animated brightness, word clouds, and keyword badges. Adjust the palettes or timings inside `generate_video_from_text` within `app.py` to experiment with other looks.
- Videos are rendered using FFmpeg through `imageio`. A standard FFmpeg build is included with the `imageio` dependency via the `imageio-ffmpeg` plugin. If you encounter codec errors, run `python -c "import imageio; imageio.plugins.ffmpeg.download()"` to fetch an updated binary.
