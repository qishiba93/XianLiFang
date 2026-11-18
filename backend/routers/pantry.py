"""
backend/routers/pantry.py
冰箱食材管理路由 (集成图库智能命中、原子级防超扣与 AI 动态实时估算减碳)
"""
import os
import uuid
import asyncio
import logging
from datetime import date, datetime, time, timedelta
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form
from fastapi.concurrency import run_in_threadpool
from openai import OpenAIError
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from backend.database import get_db
from backend.models import PantryItem, User, UserActivity, UserCarbonLog
from backend.schemas import PantryItemCreate, DeductWeightRequest, BatchDeductRequest
from backend.services.vision_service import recognize_and_decide, read_packaged_label
from backend.services.food_image_service import resolve_food_image
from backend.services.packaged_food import (
    SHELF_LABELS, decode_barcode, lookup_packaged_product, normalize_barcode, normalize_package_image,
)
from backend.services.carbon_service import (
    ai_estimate_carbon_single,
    ai_estimate_carbon_batch,
    convert_to_environmental_equivalents,
    fallback_food_category,
)
from backend.auth import get_current_user
from backend.storage import UPLOAD_DIR

router = APIRouter(prefix="/api/pantry", tags=["食材管理"])
logger = logging.getLogger(__name__)
PACKAGE_IMAGE_LIMIT = 10 * 1024 * 1024

# 1. 拍照识别 (原图先暂存)
@router.post("/recognize-food")
async def recognize_food(file: UploadFile = File(...)):
    contents = await file.read()
    ext = os.path.splitext(file.filename)[-1] or ".jpg"
    unique_filename = f"{uuid.uuid4().hex}{ext}"
    filepath = os.path.join(str(UPLOAD_DIR), unique_filename)

    with open(filepath, "wb") as f:
        f.write(contents)

    ai_decision = await recognize_and_decide(contents, file.filename)
    ai_decision["image_url"] = f"/static/uploads/{unique_filename}"
    return ai_decision


async def read_package_photo(upload: UploadFile | None) -> bytes | None:
    if upload is None:
        return None
    content = await upload.read(PACKAGE_IMAGE_LIMIT + 1)
    if len(content) > PACKAGE_IMAGE_LIMIT:
        raise HTTPException(status_code=413, detail="每张包装照片不得超过 10 MB")
    try:
        return await run_in_threadpool(normalize_package_image, content)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


def parsed_label_date(value: object) -> str | None:
    if isinstance(value, str):
        try:
            return date.fromisoformat(value).isoformat()
        except ValueError:
            pass
    return None


@router.post("/recognize-package")
async def recognize_package(
    barcode: str | None = Form(None, max_length=24),
    barcode_photo: UploadFile | None = File(None),
    label_photo: UploadFile | None = File(None),
    _: User = Depends(get_current_user),
):
    if not (barcode or barcode_photo or label_photo):
        raise HTTPException(status_code=422, detail="请填写条码或上传包装照片")
    try:
        barcode = normalize_barcode(barcode) if barcode else None
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    code_image = await read_package_photo(barcode_photo)
    label_image = await read_package_photo(label_photo)
    if not barcode and (code_image or label_image):
        barcode = await run_in_threadpool(decode_barcode, code_image or label_image)

    async def find_product():
        return await lookup_packaged_product(barcode) if barcode else {}

    async def extract_label():
        photo = label_image or code_image
        if not photo:
            return {}, False
        try:
            return await read_packaged_label(photo), False
        except (OpenAIError, ValueError):
            logger.info("Package label OCR unavailable")
            return {}, True

    catalog, (label, label_unavailable) = await asyncio.gather(find_product(), extract_label())
    label_name = label.get("food_name")
    name = catalog.get("name") or (label_name.strip()[:64] if isinstance(label_name, str) else None)
    length = label.get("shelf_life")
    unit = label.get("shelf_life_unit")
    has_duration = type(length) is int and length > 0 and unit in ("days", "months", "years")
    return {
        "barcode": barcode,
        "food_name": name or None,
        "quantity": catalog.get("quantity") or None,
        "source": "catalog" if catalog.get("name") else "label" if name else None,
        "production_date": parsed_label_date(label.get("production_date")),
        "package_expiry_date": parsed_label_date(label.get("expiry_date")),
        "package_shelf_life": length if has_duration else None,
        "package_shelf_life_unit": unit if has_duration else None,
        "label_unavailable": label_unavailable,
    }


# 2. 食材录入入库 (需求 11: 优先匹配标准图库，未匹配使用实拍图)
@router.post("/items")
async def create_pantry_item(
    item_in: PantryItemCreate,
    db: AsyncSession = Depends(get_db),
    user: User = Depends(get_current_user)
):
    if item_in.package_expiry_date:
        expire_at = datetime.combine(item_in.package_expiry_date, time(23, 59, 59))
    else:
        expire_at = datetime.now() + timedelta(days=item_in.shelf_life_days)
    rem_weight = item_in.remaining_weight if item_in.remaining_weight is not None else item_in.initial_weight

    # 智能判图决策 (需求 11)
    img_decision = await resolve_food_image(
        db=db,
        food_name=item_in.name,
        user_uploaded_url=item_in.image_url
    )

    item = PantryItem(
        user_id=user.id,
        name=item_in.name,
        location=item_in.location,
        shelf=item_in.shelf,
        barcode=item_in.barcode,
        initial_weight=item_in.initial_weight,
        remaining_weight=rem_weight,
        unit=item_in.unit,
        expire_at=expire_at,
        image_url=img_decision.final_url
    )
    db.add(item)
    db.add(UserActivity(user_id=user.id, activity_type="add", note=f"录入食材:{item.name}"))
    await db.commit()
    await db.refresh(item)
    return item

# 3. 食材列表查询
@router.get("/items")
async def list_pantry_items(
    db: AsyncSession = Depends(get_db),
    user: User = Depends(get_current_user)
):
    stmt = (
        select(PantryItem)
        .where(PantryItem.user_id == user.id, PantryItem.remaining_weight > 0)
        .order_by(PantryItem.expire_at.asc())
    )
    result = await db.execute(stmt)
    items = result.scalars().all()

    now = datetime.now()
    resp = []
    for item in items:
        diff_hours = (item.expire_at - now).total_seconds() / 3600.0
        resp.append({
            "id": item.id,
            "name": item.name,
            "location": item.location,
            "shelf": item.shelf,
            "shelf_label": SHELF_LABELS.get(item.location, {}).get(item.shelf),
            "barcode": item.barcode,
            "remaining_weight": item.remaining_weight,
            "initial_weight": item.initial_weight,
            "unit": item.unit,
            "expire_at": item.expire_at.strftime("%Y-%m-%d %H:%M:%S"),
            "image_url": item.image_url,
            "is_expired": diff_hours <= 0,
            "hours_left": round(diff_hours, 1)
        })
    return resp

# 4. 单食材消耗扣减 (需求 12: 由 AI 动态智能生成该次消耗的减碳克数)
@router.post("/items/{item_id}/deduct")
async def deduct_pantry_item(
    item_id: int,
    req: DeductWeightRequest,
    db: AsyncSession = Depends(get_db),
    user: User = Depends(get_current_user)
):
    stmt = select(PantryItem).where(PantryItem.id == item_id, PantryItem.user_id == user.id).with_for_update()
    result = await db.execute(stmt)
    item = result.scalar_one_or_none()

    if not item:
        raise HTTPException(status_code=404, detail="未找到该食材")
    if item.remaining_weight < req.deduct_weight:
        raise HTTPException(status_code=400, detail=f"【库存不足】当前仅剩 {item.remaining_weight} {item.unit}，严禁超扣！")

    recipe_name = req.recipe_name or "家常美味"

    # 由 AI 实时推演该次做菜的真实减碳贡献与理由
    if item.unit in ("毫升", "瓶"):
        carbon_saved, ai_reason, food_category = 0.0, "已记入用量账本", fallback_food_category(item.name)
    else:
        carbon_saved, ai_reason, food_category = await ai_estimate_carbon_single(
            food_name=item.name,
            weight_grams=req.deduct_weight * (500 if item.unit == "斤" else 1),
            recipe_name=recipe_name
        )

    carbon_log = UserCarbonLog(
        user_id=user.id,
        item_name=item.name,
        weight_grams=req.deduct_weight,
        item_unit=item.unit,
        carbon_saved_grams=carbon_saved,
        food_category=food_category,
        source_recipe=recipe_name
    )
    db.add(carbon_log)
    db.add(UserActivity(user_id=user.id, activity_type="consume", note=f"制作菜品消耗:{item.name}"))

    remaining_after = round(item.remaining_weight - req.deduct_weight, 2)
    is_used_up = (remaining_after <= 0)

    if is_used_up:
        item_name = item.name
        await db.delete(item)
    else:
        item.remaining_weight = remaining_after

    await db.commit()

    return {
        "message": "用量已记入账本，库存已更新",
        "ai_reason": ai_reason,
        "is_used_up": is_used_up,
        "remaining_weight": max(0.0, remaining_after),
        "carbon_saved_grams": carbon_saved,
        "food_category": food_category,
        "equivalents": convert_to_environmental_equivalents(carbon_saved)
    }

# 5. 批量多食材核对扣减 (需求 12: 由 AI 一次性深度打包推算全单减碳与绿色洞察)
@router.post("/items/batch-deduct")
async def batch_deduct_pantry_items(
    req: BatchDeductRequest,
    db: AsyncSession = Depends(get_db),
    user: User = Depends(get_current_user)
):
    if not req.items:
        raise HTTPException(status_code=400, detail="扣减清单为空")

    if len({entry.item_id for entry in req.items}) != len(req.items):
        raise HTTPException(status_code=400, detail="每个库存批次只能核销一次")

    updated_items = []
    deleted_names = []
    items_to_process = []

    try:
        # 第一步：排他锁校验库存，确保没有任何一项出现超扣
        for deduct_info in req.items:
            stmt = (
                select(PantryItem)
                .where(PantryItem.id == deduct_info.item_id, PantryItem.user_id == user.id)
                .with_for_update()
            )
            res = await db.execute(stmt)
            item = res.scalar_one_or_none()

            if not item:
                raise HTTPException(status_code=404, detail=f"食材 ID {deduct_info.item_id} 已不在冰箱中")

            if deduct_info.deduct_weight > item.remaining_weight:
                raise HTTPException(
                    status_code=400,
                    detail=f"【严禁超扣】食材「{item.name}」尝试扣减 {deduct_info.deduct_weight}{item.unit}，但数据库仅存 {item.remaining_weight}{item.unit}！整个核减事务已全部取消。"
                )

            items_to_process.append({
                "item_orm": item,
                "id": item.id,
                "name": item.name,
                "weight": deduct_info.deduct_weight,
                "unit": item.unit
            })

        # 第二步：将所有待扣减食材打包交给 AI 实时计算综合减碳
        ai_batch_payload = [
            {"id": it["id"], "name": it["name"], "weight": it["weight"] * (500 if it["unit"] == "斤" else 1)}
            for it in items_to_process if it["unit"] not in ("毫升", "瓶")
        ]
        total_carbon, breakdown_map, category_map, eco_insight = 0.0, {}, {}, ""
        if ai_batch_payload:
            total_carbon, breakdown_map, category_map, eco_insight = await ai_estimate_carbon_batch(
                items=ai_batch_payload,
                recipe_name=req.recipe_name
            )

        # 第三步：执行物理扣减并写入减碳明细账本
        for it in items_to_process:
            item = it["item_orm"]
            deduct_weight = it["weight"]
            item_carbon = breakdown_map.get(item.id, 0.0)
            food_category = category_map.get(item.id, fallback_food_category(item.name))

            db.add(UserCarbonLog(
                user_id=user.id,
                item_name=item.name,
                weight_grams=deduct_weight,
                item_unit=it["unit"],
                carbon_saved_grams=item_carbon,
                food_category=food_category,
                source_recipe=req.recipe_name
            ))

            new_weight = round(item.remaining_weight - deduct_weight, 2)
            if new_weight <= 0:
                deleted_names.append(item.name)
                await db.delete(item)
            else:
                item.remaining_weight = new_weight
                updated_items.append({"name": item.name, "remaining": new_weight, "unit": item.unit})

        db.add(UserActivity(user_id=user.id, activity_type="consume", note=f"批量制作菜谱:{req.recipe_name}"))
        await db.commit()

    except Exception:
        await db.rollback()
        raise

    return {
        "message": f"批量核销完成！AI 估算本次为地球减排约 {round(total_carbon, 1)}g CO2e！",
        "eco_insight": eco_insight,
        "total_carbon_saved_grams": round(total_carbon, 1),
        "category_totals": {
            "vegetarian": round(sum(breakdown_map.get(it["id"], 0.0) for it in items_to_process if category_map.get(it["id"], fallback_food_category(it["name"])) == "vegetarian"), 1),
            "non_vegetarian": round(sum(breakdown_map.get(it["id"], 0.0) for it in items_to_process if category_map.get(it["id"], fallback_food_category(it["name"])) == "non_vegetarian"), 1),
        },
        "equivalents": convert_to_environmental_equivalents(total_carbon),
        "updated": updated_items,
        "cleared": deleted_names
    }

# 6. 手动丢弃删除
@router.post("/items/{item_id}/discard")
@router.delete("/items/{item_id}")
async def discard_pantry_item(item_id: int, db: AsyncSession = Depends(get_db), user: User = Depends(get_current_user)):
    stmt = select(PantryItem).where(PantryItem.id == item_id, PantryItem.user_id == user.id)
    result = await db.execute(stmt)
    item = result.scalar_one_or_none()
    if not item:
        raise HTTPException(status_code=404, detail="食材不存在或已清除")

    await db.delete(item)
    await db.commit()
    return {"message": f"食材「{item.name}」已从冰箱中彻底移除"}
