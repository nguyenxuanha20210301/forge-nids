"""saves every figure in the three formats the project uses.

    <name>.png      raster at 600 dpi
    <name>.pdf      vector, sized for the paper
    <name>_a4.pdf   the same vector figure scaled to fill one A4 page
"""

from io import BytesIO
from pathlib import Path

import matplotlib.pyplot as plt
from pypdf import PdfReader, PdfWriter, Transformation

A4_POINTS = (595.276, 841.89)
MARGIN_POINTS = 42.52  # 15 mm


def _a4_page(pdf_bytes: bytes, out_path: Path) -> None:
    src = PdfReader(BytesIO(pdf_bytes)).pages[0]
    w, h = float(src.mediabox.width), float(src.mediabox.height)
    # wide figures go on a landscape page, tall ones on a portrait page
    page_w, page_h = A4_POINTS if h >= w else A4_POINTS[::-1]
    scale = min((page_w - 2 * MARGIN_POINTS) / w, (page_h - 2 * MARGIN_POINTS) / h)
    tx = (page_w - w * scale) / 2 - float(src.mediabox.left) * scale
    ty = (page_h - h * scale) / 2 - float(src.mediabox.bottom) * scale
    writer = PdfWriter()
    page = writer.add_blank_page(width=page_w, height=page_h)
    page.merge_transformed_page(src, Transformation().scale(scale).translate(tx, ty))
    with open(out_path, "wb") as f:
        writer.write(f)


def save_figure(fig, folder, name: str, close: bool = True) -> list[Path]:
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    if fig._suptitle is not None or any(ax.get_title() for ax in fig.axes):
        raise ValueError(f"figure {name} has a title, titles belong in the caption of the paper")
    png = folder / f"{name}.png"
    pdf = folder / f"{name}.pdf"
    a4 = folder / f"{name}_a4.pdf"
    fig.savefig(png, dpi=600)
    buf = BytesIO()
    fig.savefig(buf, format="pdf")
    pdf.write_bytes(buf.getvalue())
    _a4_page(buf.getvalue(), a4)
    if close:
        plt.close(fig)
    return [png, pdf, a4]
