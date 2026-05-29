from pathlib import Path


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