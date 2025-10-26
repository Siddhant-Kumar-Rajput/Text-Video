import os
import textwrap
import uuid
from datetime import datetime
from typing import List

import imageio
import numpy as np
from flask import Flask, render_template, request
from PIL import Image, ImageDraw, ImageFont

app = Flask(__name__)

VIDEOS_DIR = os.path.join(app.static_folder, "videos")
os.makedirs(VIDEOS_DIR, exist_ok=True)


def _load_font(base_size: int = 32) -> ImageFont.FreeTypeFont:
    """Load a readable font, falling back to the default when needed."""
    try:
        return ImageFont.truetype("DejaVuSans.ttf", base_size)
    except OSError:
        return ImageFont.load_default()


def _wrap_text(text: str, max_chars: int = 36) -> List[str]:
    """Wrap long text into multiple lines that fit the frame width."""
    wrapped_lines = []
    for paragraph in text.splitlines():
        paragraph = paragraph.strip()
        if not paragraph:
            wrapped_lines.append("")
            continue
        wrapped_lines.extend(textwrap.wrap(paragraph, width=max_chars))
    return wrapped_lines or [""]


def generate_video_from_text(text: str, output_path: str, fps: int = 24) -> None:
    """Render a simple 2D video slideshow that highlights each wrapped line."""
    font = _load_font()
    width, height = 960, 540
    margin_x, margin_y = 60, 60
    background_color = (15, 23, 42)  # dark blue tone
    text_color = (226, 232, 240)  # light gray
    highlight_color = (96, 165, 250)  # accent for the active line

    lines = _wrap_text(text)
    line_height = font.getbbox("Ag")[3] - font.getbbox("Ag")[1] + 12
    total_display_seconds_per_line = 2.5
    frames_per_line = max(1, int(total_display_seconds_per_line * fps))

    frames: List[np.ndarray] = []
    total_text_height = line_height * len(lines)
    start_y = max(margin_y, (height - total_text_height) // 2)

    for index, line in enumerate(lines):
        for frame_index in range(frames_per_line):
            img = Image.new("RGB", (width, height), color=background_color)
            draw = ImageDraw.Draw(img)

            # Add a subtle progress bar at the bottom.
            progress_ratio = ((index * frames_per_line) + frame_index + 1) / (len(lines) * frames_per_line)
            progress_width = int(progress_ratio * (width - 2 * margin_x))
            progress_bar_height = 10
            progress_bar_y = height - margin_y
            draw.rectangle(
                [
                    (margin_x, progress_bar_y),
                    (margin_x + progress_width, progress_bar_y + progress_bar_height),
                ],
                fill=highlight_color,
            )

            # Draw the wrapped text lines.
            for line_idx, current_line in enumerate(lines):
                y = start_y + line_idx * line_height
                color = highlight_color if line_idx == index else text_color
                draw.text((margin_x, y), current_line, font=font, fill=color)

            frames.append(np.array(img))

    # Keep the final frame on screen for a moment.
    final_frame_count = fps * 2
    for _ in range(final_frame_count):
        frames.append(frames[-1])

    writer = imageio.get_writer(
        output_path,
        fps=fps,
        codec="libx264",
        format="FFMPEG",
        macro_block_size=None,
    )
    try:
        for frame in frames:
            writer.append_data(frame)
    finally:
        writer.close()


@app.route("/", methods=["GET", "POST"])
def index():
    video_filename = None
    error = None
    if request.method == "POST":
        story_text = request.form.get("story", "").strip()
        if not story_text:
            error = "Please provide some text to generate a video."
        else:
            timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
            token = uuid.uuid4().hex[:6]
            filename = f"text_video_{timestamp}_{token}.mp4"
            output_path = os.path.join(VIDEOS_DIR, filename)
            try:
                generate_video_from_text(story_text, output_path)
                video_filename = filename
            except Exception as exc:  # pragma: no cover - user feedback
                error = f"Failed to generate video: {exc}"
                if os.path.exists(output_path):
                    os.remove(output_path)

    return render_template("index.html", video_filename=video_filename, error=error)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
