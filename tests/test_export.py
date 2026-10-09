import matplotlib.pyplot as plt
import pytest
from pypdf import PdfReader

from forge.viz.export import A4_POINTS, save_figure


def test_three_formats(tmp_path):
    fig, ax = plt.subplots(figsize=(3.5, 2.4))
    ax.plot([0, 1], [0, 1])
    png, pdf, a4 = save_figure(fig, tmp_path, "line")
    assert png.stat().st_size > 0 and pdf.stat().st_size > 0
    page = PdfReader(a4).pages[0]
    size = sorted([float(page.mediabox.width), float(page.mediabox.height)])
    assert size == pytest.approx(sorted(A4_POINTS), abs=0.5)


def test_title_rejected(tmp_path):
    fig, ax = plt.subplots()
    ax.set_title("no")
    with pytest.raises(ValueError):
        save_figure(fig, tmp_path, "titled")
