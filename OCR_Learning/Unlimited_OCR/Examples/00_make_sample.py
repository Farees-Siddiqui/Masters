"""Render a synthetic document image so the other examples have test data.

Writes images/sample.png and images/sample_truth.txt (ground truth for CER/WER
in 03_compare_models.py). No OCR engine required.
"""

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

HERE = Path(__file__).parent
IMAGES = HERE / "images"

TITLE = "Quarterly Energy Report"
BODY = [
    "Solar generation rose 14% year over year, driven by new capacity",
    "in the southwest region. Wind output was flat, while natural gas",
    "declined 6% as older plants were retired ahead of schedule.",
    "",
    "Region      Solar    Wind     Gas",
    "North        120      340     510",
    "South        480      210     390",
    "West         610      180     220",
    "",
    "Full methodology is described in Appendix B. Figures are given in",
    "gigawatt hours and rounded to the nearest whole number.",
]


def load_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    for name in ("arial.ttf", "calibri.ttf", "DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def main() -> None:
    IMAGES.mkdir(exist_ok=True)
    img = Image.new("RGB", (1200, 800), "white")
    draw = ImageDraw.Draw(img)

    draw.text((60, 50), TITLE, fill="black", font=load_font(44))
    body_font = load_font(28)
    y = 140
    for line in BODY:
        draw.text((60, y), line, fill="black", font=body_font)
        y += 46

    img.save(IMAGES / "sample.png")
    truth = TITLE + "\n" + "\n".join(BODY) + "\n"
    (IMAGES / "sample_truth.txt").write_text(truth, encoding="utf-8")
    print(f"Wrote {IMAGES / 'sample.png'} and {IMAGES / 'sample_truth.txt'}")


if __name__ == "__main__":
    main()
