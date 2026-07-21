from pathlib import Path

import fitz


def to_upload_url(file_path: str | None) -> str | None:
    """
    app/uploads/... yoki app\\uploads\\... pathni frontend ko‘ra oladigan URL ga aylantiradi.

    Masalan:
    app\\uploads\\results\\abc.png -> /uploads/results/abc.png
    """
    if not file_path:
        return None

    normalized = str(file_path).replace("\\", "/")

    marker = "app/uploads/"
    if marker in normalized:
        relative = normalized.split(marker, 1)[1]
        return f"/uploads/{relative}"

    if normalized.startswith("uploads/"):
        return f"/{normalized}"

    if normalized.startswith("/uploads/"):
        return normalized

    return normalized


def ensure_file_preview(file_path: str | None) -> str | None:
    """
    Rasm fayli bo‘lsa o‘zini qaytaradi.
    PDF bo‘lsa birinchi sahifasini PNG previewga aylantiradi.
    """
    if not file_path:
        return None

    source_path = Path(file_path)

    if source_path.suffix.lower() != ".pdf":
        return str(source_path)

    if not source_path.exists():
        return None

    preview_path = source_path.with_name(
        f"{source_path.stem}_preview.png"
    )

    if preview_path.exists():
        return str(preview_path)

    try:
        preview_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        with fitz.open(str(source_path)) as document:
            if document.page_count == 0:
                return None

            page = document.load_page(0)
            matrix = fitz.Matrix(2.0, 2.0)

            pixmap = page.get_pixmap(
                matrix=matrix,
                alpha=False,
            )

            pixmap.save(str(preview_path))

    except Exception:
        # Preview xatosi natijaning o‘zini buzmasligi kerak.
        return None

    return str(preview_path)