import os
import random
import re
import uuid
from collections import Counter
from datetime import datetime
from typing import List, Optional

import imageio
import numpy as np
from flask import Flask, render_template, request
from PIL import Image, ImageDraw, ImageEnhance, ImageFont
from wordcloud import WordCloud

app = Flask(__name__)

VIDEOS_DIR = os.path.join(app.static_folder, "videos")
os.makedirs(VIDEOS_DIR, exist_ok=True)

FONT_CANDIDATES = [
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/local/share/fonts/DejaVuSans.ttf",
    "DejaVuSans.ttf",
]

STOPWORDS = {
    "the",
    "and",
    "that",
    "with",
    "from",
    "into",
    "this",
    "have",
    "your",
    "about",
    "there",
    "their",
    "would",
    "could",
    "should",
    "while",
    "where",
    "which",
    "when",
    "ours",
    "they",
    "them",
    "over",
    "some",
    "being",
    "also",
    "just",
    "into",
    "like",
    "were",
    "been",
    "each",
    "more",
    "than",
    "ever",
    "only",
    "through",
    "because",
    "very",
    "such",
    "every",
    "those",
}

PALETTES = [
    ((15, 23, 42), (30, 64, 175), (96, 165, 250)),
    ((17, 24, 39), (59, 130, 246), (148, 163, 253)),
    ((13, 42, 63), (17, 94, 89), (45, 212, 191)),
    ((39, 24, 126), (99, 102, 241), (165, 180, 252)),
    ((76, 29, 149), (139, 92, 246), (233, 213, 255)),
]


def _get_font_path() -> Optional[str]:
    for path in FONT_CANDIDATES:
        if os.path.exists(path):
            return path
    return None


def _load_font(base_size: int = 32) -> ImageFont.FreeTypeFont:
    """Load a readable font, falling back to the default when needed."""
    font_path = _get_font_path()
    if font_path:
        try:
            return ImageFont.truetype(font_path, base_size)
        except OSError:
            pass
    return ImageFont.load_default()


def _split_sentences(text: str) -> List[str]:
    cleaned = re.sub(r"\s+", " ", text).strip()
    if not cleaned:
        return []
    potential_sentences = re.split(r"(?<=[.!?])\s+|\n+", cleaned)
    sentences = [sentence.strip() for sentence in potential_sentences if sentence.strip()]
    return sentences or [cleaned]


def _wrap_line_by_width(text: str, font: ImageFont.ImageFont, max_width: int) -> List[str]:
    words = text.split()
    if not words:
        return [""]

    lines: List[str] = []
    current_words: List[str] = []

    for word in words:
        trial = " ".join(current_words + [word]) if current_words else word
        if font.getlength(trial) <= max_width:
            current_words.append(word)
        else:
            if current_words:
                lines.append(" ".join(current_words))
            current_words = [word]
    if current_words:
        lines.append(" ".join(current_words))
    return lines


def _extract_keywords(text: str, top_n: int = 5) -> List[str]:
    tokens = re.findall(r"[A-Za-z']+", text.lower())
    filtered = [token for token in tokens if token not in STOPWORDS and len(token) > 2]
    if not filtered:
        return []
    counts = Counter(filtered)
    return [word for word, _ in counts.most_common(top_n)]


def _create_gradient(width: int, height: int, start_color: tuple[int, int, int], end_color: tuple[int, int, int]) -> Image.Image:
    gradient = np.zeros((height, width, 3), dtype=np.float32)
    for index in range(3):
        channel = np.linspace(start_color[index], end_color[index], height, dtype=np.float32)
        gradient[:, :, index] = np.tile(channel[:, None], (1, width))
    return Image.fromarray(np.clip(gradient, 0, 255).astype(np.uint8), mode="RGB")


def _color_blend(color_a: tuple[int, int, int], color_b: tuple[int, int, int], amount: float) -> tuple[int, int, int]:
    return tuple(int(color_a[i] * (1 - amount) + color_b[i] * amount) for i in range(3))


def _generate_wordcloud(
    text: str,
    width: int,
    height: int,
    palette: tuple[tuple[int, int, int], ...],
    font_path: Optional[str],
) -> Image.Image:
    keywords = text if text.strip() else "visual"

    def color_func(*_: int, **__: int) -> str:
        base = random.random()
        color = _color_blend(palette[1], palette[2], base)
        return f"rgb({color[0]}, {color[1]}, {color[2]})"

    cloud = WordCloud(
        width=width,
        height=height,
        mode="RGBA",
        background_color=None,
        stopwords=STOPWORDS,
        max_words=120,
        color_func=color_func,
        font_path=font_path,
    ).generate(keywords)
    return Image.fromarray(cloud.to_array())


def _build_background(
    sentence: str,
    width: int,
    height: int,
    palette: tuple[tuple[int, int, int], ...],
    font_path: Optional[str],
    fallback_tokens: List[str],
) -> Image.Image:
    gradient = _create_gradient(width, height, palette[0], palette[1]).convert("RGBA")

    cloud_text = sentence
    if len(sentence.split()) < 4 and fallback_tokens:
        cloud_text = sentence + " " + " ".join(fallback_tokens)

    try:
        cloud_image = _generate_wordcloud(cloud_text, width, height, palette, font_path)
    except ValueError:
        cloud_image = Image.new("RGBA", (width, height), (0, 0, 0, 0))

    scale = 0.92
    resized = cloud_image.resize((int(width * scale), int(height * scale)), Image.LANCZOS)
    position = (int((width - resized.width) / 2), int((height - resized.height) / 2))

    composed = gradient.copy()
    composed.alpha_composite(resized, dest=position)

    vignette = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    vignette_draw = ImageDraw.Draw(vignette)
    vignette_draw.rectangle([0, 0, width, height], fill=(8, 15, 40, 110))
    composed.alpha_composite(vignette)

    shimmer = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    shimmer_draw = ImageDraw.Draw(shimmer)
    shimmer_draw.ellipse(
        [
            int(width * 0.05),
            int(height * 0.25),
            int(width * 0.55),
            int(height * 0.85),
        ],
        fill=(*palette[2], 40),
    )
    composed.alpha_composite(shimmer)

    return composed


def _draw_scene_label(
    draw: ImageDraw.ImageDraw,
    index: int,
    total: int,
    font: ImageFont.ImageFont,
    width: int,
    accent: tuple[int, int, int],
) -> None:
    label = f"Scene {index + 1} of {total}"
    bbox = font.getbbox(label)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]
    padding_x, padding_y = 18, 12
    x = width - text_width - padding_x * 2 - 80
    y = 60
    box = [x, y, x + text_width + padding_x * 2, y + text_height + padding_y * 2]
    draw.rounded_rectangle(box, radius=20, fill=(*accent, 160))
    draw.text((x + padding_x, y + padding_y), label, font=font, fill=(15, 23, 42))


def _draw_sentence_block(
    draw: ImageDraw.ImageDraw,
    sentence: str,
    font: ImageFont.ImageFont,
    width: int,
    height: int,
    accent: tuple[int, int, int],
) -> None:
    max_text_width = width - 280
    lines = _wrap_line_by_width(sentence, font, max_text_width)
    metrics = font.getbbox("Ag")
    line_height = (metrics[3] - metrics[1]) + 12
    text_height = line_height * len(lines)
    start_y = max(140, int((height - text_height) / 2))
    text_width = max((font.getlength(line) for line in lines), default=0)
    padding_x, padding_y = 48, 36
    rect = [
        140,
        start_y - padding_y,
        140 + text_width + padding_x,
        start_y + text_height + padding_y,
    ]
    draw.rounded_rectangle(rect, radius=32, fill=(15, 23, 42, 200), outline=(*accent, 160), width=2)

    for offset, line in enumerate(lines):
        y = start_y + offset * line_height
        draw.text((160, y), line, font=font, fill=(226, 232, 240))


def _draw_keyword_badges(
    draw: ImageDraw.ImageDraw,
    keywords: List[str],
    font: ImageFont.ImageFont,
    width: int,
    height: int,
    accent: tuple[int, int, int],
    secondary: tuple[int, int, int],
) -> None:
    if not keywords:
        return

    margin_x = 160
    base_y = height - 220
    x = margin_x
    y = base_y
    gap = 20

    caption = "Key ideas"
    draw.text((margin_x, base_y - 42), caption, font=font, fill=(*accent, 220))

    for keyword in keywords:
        text = keyword.capitalize()
        bbox = font.getbbox(text)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
        padding_x, padding_y = 22, 14
        box = [
            x,
            y,
            x + text_width + padding_x * 2,
            y + text_height + padding_y * 2,
        ]
        draw.rounded_rectangle(box, radius=28, fill=(*secondary, 180))
        draw.text((x + padding_x, y + padding_y), text, font=font, fill=(15, 23, 42))
        x = box[2] + gap
        if x > width - margin_x:
            x = margin_x
            y += text_height + padding_y * 2 + gap


def _draw_progress_bar(
    draw: ImageDraw.ImageDraw,
    sentence_index: int,
    frame_index: int,
    frames_per_sentence: int,
    total_sentences: int,
    width: int,
    height: int,
    accent: tuple[int, int, int],
) -> None:
    total_frames = max(1, total_sentences * frames_per_sentence)
    current_frame = sentence_index * frames_per_sentence + frame_index + 1
    ratio = min(1.0, current_frame / total_frames)

    bar_width = width - 320
    bar_height = 18
    x0 = 160
    y0 = height - 120

    background_color = (226, 232, 240, 90)
    draw.rounded_rectangle([x0, y0, x0 + bar_width, y0 + bar_height], radius=bar_height // 2, fill=background_color)

    fill_width = max(0, int(bar_width * ratio))
    if fill_width:
        draw.rounded_rectangle(
            [x0, y0, x0 + fill_width, y0 + bar_height],
            radius=bar_height // 2,
            fill=(*accent, 220),
        )


def _animate_frame(frame: Image.Image, intensity: float) -> Image.Image:
    enhancer = ImageEnhance.Brightness(frame)
    return enhancer.enhance(0.96 + 0.08 * intensity)


def generate_video_from_text(text: str, output_path: str, fps: int = 24) -> None:
    """Render a visually rich slideshow video for the provided text."""
    sentences = _split_sentences(text)
    if not sentences:
        sentences = [text]

    width, height = 1280, 720
    headline_font = _load_font(54)
    label_font = _load_font(28)
    badge_font = _load_font(26)
    frames: List[np.ndarray] = []

    seconds_per_scene = 3.5
    frames_per_scene = max(1, int(seconds_per_scene * fps))

    font_path = _get_font_path()
    story_keywords = _extract_keywords(text, top_n=18)

    for index, sentence in enumerate(sentences):
        palette = PALETTES[index % len(PALETTES)]
        background = _build_background(sentence, width, height, palette, font_path, story_keywords)
        for frame_index in range(frames_per_scene):
            frame = background.copy()
            intensity = frame_index / max(1, frames_per_scene - 1)
            animated_frame = _animate_frame(frame, intensity)
            draw = ImageDraw.Draw(animated_frame, "RGBA")

            _draw_scene_label(draw, index, len(sentences), label_font, width, palette[2])
            _draw_sentence_block(draw, sentence, headline_font, width, height, palette[2])

            keywords = _extract_keywords(sentence, top_n=4)
            if not keywords:
                keywords = story_keywords[:4]
            _draw_keyword_badges(draw, keywords, badge_font, width, height, palette[2], palette[1])
            _draw_progress_bar(draw, index, frame_index, frames_per_scene, len(sentences), width, height, palette[2])

            frames.append(np.array(animated_frame.convert("RGB")))

    if frames:
        final_hold = frames[-1]
        frames.extend([final_hold] * (fps * 2))

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
