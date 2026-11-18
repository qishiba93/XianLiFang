"""Barcode lookup and deterministic dates from a package label."""
from calendar import monthrange
from datetime import date, timedelta
from io import BytesIO
import logging
import re

import httpx
from PIL import Image, ImageOps, UnidentifiedImageError

logger = logging.getLogger(__name__)

SHELF_LABELS = {
    "refrigeration": {"top": "冷藏顶层", "middle": "冷藏中层", "bottom": "冷藏底层"},
    "freezer": {"top": "冷冻上层", "bottom": "冷冻底层"},
}


def normalize_barcode(value: str) -> str:
    code = re.sub(r"\s+", "", value)
    if not re.fullmatch(r"(?:[0-9]{8}|[0-9]{12,14})", code):
        raise ValueError("请输入 8、12、13 或 14 位商品条码")
    checksum = sum(int(digit) * (3 if i % 2 == 0 else 1)
                   for i, digit in enumerate(reversed(code[:-1])))
    if (checksum + int(code[-1])) % 10:
        raise ValueError("条码校验失败，请核对包装上的数字")
    return code


def calculate_expiry(produced: date, length: int, unit: str) -> date:
    if unit == "days":
        return produced + timedelta(days=length)
    months = length * 12 if unit == "years" else length
    year, month_offset = divmod(produced.year * 12 + produced.month - 1 + months, 12)
    month = month_offset + 1
    return date(year, month, min(produced.day, monthrange(year, month)[1]))


def normalize_package_image(content: bytes) -> bytes:
    try:
        with Image.open(BytesIO(content)) as image:
            if image.format not in {"JPEG", "PNG", "WEBP"} or image.width * image.height > 24_000_000:
                raise ValueError("请上传 JPEG、PNG 或 WebP 图片，像素总数不得超过 2400 万")
            image = ImageOps.exif_transpose(image)
            image.thumbnail((2400, 2400))
            normalized = BytesIO()
            image.convert("RGB").save(normalized, "JPEG", quality=92)
            return normalized.getvalue()
    except (UnidentifiedImageError, Image.DecompressionBombError, OSError) as exc:
        raise ValueError("图片无法读取，请重新上传包装照片") from exc


def decode_barcode(content: bytes) -> str | None:
    # 仅在识别包装条码时加载，其他接口无需初始化 OpenCV。
    import cv2
    import numpy as np

    image = cv2.imdecode(np.frombuffer(content, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
    if image is None:
        return None
    detector = cv2.barcode_BarcodeDetector()

    def valid_code(candidates) -> str | None:
        for candidate in candidates:
            try:
                return normalize_barcode(candidate)
            except ValueError:
                continue
        return None

    try:
        _, decoded, _, _ = detector.detectAndDecodeWithType(image)
        code = valid_code(decoded)
        if code:
            return code
        # 条码特写可能没有被定位为独立区域，直接读取裁好的图片。
        for view in (image, cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)):
            h, w = view.shape
            if min(h, w) < 24:
                continue
            corners = np.array([[[0, h - 1], [0, 0], [w - 1, 0], [w - 1, h - 1]]], dtype=np.float32)
            _, decoded, _ = detector.decodeWithType(view, corners)
            code = valid_code(decoded)
            if code:
                return code
    except cv2.error:
        logger.info("Barcode image could not be decoded")
    return None


async def lookup_packaged_product(barcode: str) -> dict:
    fields = "product_name_zh,product_name,brands,quantity"
    headers = {"User-Agent": "XianLiFang/1.0 (https://github.com/qishiba93/XianLiFang)"}
    try:
        async with httpx.AsyncClient(timeout=4, headers=headers) as client:
            response = await client.get(
                f"https://world.openfoodfacts.org/api/v2/product/{barcode}.json",
                params={"fields": fields},
            )
            if response.status_code == 404:
                return {}
            response.raise_for_status()
            product = response.json().get("product") or {}
    except (httpx.HTTPError, ValueError):
        logger.info("Package product catalog lookup unavailable")
        return {}

    name = product.get("product_name_zh") or product.get("product_name")
    return {
        "name": str(name)[:64] if name else None,
        "quantity": str(product.get("quantity") or "")[:64],
    }
