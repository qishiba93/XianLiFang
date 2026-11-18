import unittest
from unittest.mock import patch
from datetime import datetime, timedelta
from urllib.parse import quote

from httpx import ASGITransport, AsyncClient
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from backend.auth import create_access_token
from backend.database import Base, get_db
from backend.main import app
from backend.models import (
    CommunityPost,
    FavoriteRecipe,
    PantryItem,
    PostComment,
    ShoppingItem,
    StandardFoodImage,
    User,
    UserCarbonLog,
)
from backend.routers.carbon import get_current_week_start


class ButtonRouteTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.engine = create_async_engine("sqlite+aiosqlite:///:memory:")
        self.sessions = async_sessionmaker(self.engine, expire_on_commit=False)
        async with self.engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)

        async def test_db():
            async with self.sessions() as session:
                yield session

        app.dependency_overrides[get_db] = test_db
        self.client = AsyncClient(transport=ASGITransport(app=app), base_url="http://test")

        async with self.sessions() as db:
            admin = User(username="route-admin", hashed_password="unused", role="super_admin")
            owner = User(username="route-owner", hashed_password="unused", role="user")
            other = User(username="route-other", hashed_password="unused", role="user")
            db.add_all([admin, owner, other])
            await db.flush()
            self.headers = {
                user.username: {"X-FreshFood-Token": create_access_token({"sub": user.username})}
                for user in (admin, owner, other)
            }

            self.favorite_name = "番茄/鸡蛋"
            favorite = FavoriteRecipe(
                user_id=owner.id, recipe_name=self.favorite_name,
                ingredients_needed=[], cooking_steps=[],
            )
            favorite_by_id = FavoriteRecipe(
                user_id=owner.id, recipe_name="按 ID 删除",
                ingredients_needed=[], cooking_steps=[],
            )
            pantry = PantryItem(
                user_id=owner.id, name="过期番茄", location="refrigeration",
                initial_weight=100, remaining_weight=100, unit="克",
                expire_at=datetime.now() - timedelta(days=1),
            )
            shopping = ShoppingItem(user_id=owner.id, name="葱")
            post = CommunityPost(
                user_id=owner.id, title="测试动态", content="本地测试",
                image_url="/static/standards/tomato_1.jpg", status=1,
            )
            image = StandardFoodImage(
                food_name="测试标准图", category="vegetable",
                image_url="/static/standards/tomato_1.jpg",
            )
            db.add_all([favorite, favorite_by_id, pantry, shopping, post, image])
            await db.flush()
            comment = PostComment(post_id=post.id, user_id=owner.id, content="测试评论", status=1)
            db.add(comment)
            await db.flush()
            self.ids = {
                "favorite": favorite.id,
                "favorite_by_id": favorite_by_id.id,
                "pantry": pantry.id,
                "shopping": shopping.id,
                "post": post.id,
                "comment": comment.id,
                "image": image.id,
            }
            await db.commit()

    async def asyncTearDown(self):
        await self.client.aclose()
        app.dependency_overrides.clear()
        await self.engine.dispose()

    async def test_post_delete_buttons_reach_authorized_handlers(self):
        owner = self.headers["route-owner"]
        other = self.headers["route-other"]
        admin = self.headers["route-admin"]
        favorite_url = f"/api/recipes/favorites/{self.ids['favorite_by_id']}/delete"

        self.assertEqual((await self.client.post(favorite_url)).status_code, 401)
        self.assertEqual((await self.client.post(favorite_url, headers=other)).status_code, 404)
        self.assertEqual((await self.client.post(favorite_url, headers=owner)).status_code, 200)
        self.assertEqual((await self.client.post(favorite_url, headers=owner)).status_code, 404)

        paths = [
            f"/api/recipes/favorites/by-name/{quote(self.favorite_name, safe='')}/delete",
            f"/api/pantry/items/{self.ids['pantry']}/discard",
            f"/api/shopping/items/{self.ids['shopping']}/delete",
            f"/api/community/comments/{self.ids['comment']}/delete",
            f"/api/community/posts/{self.ids['post']}/delete",
        ]
        for path in paths:
            with self.subTest(path=path):
                response = await self.client.post(path, headers=owner)
                self.assertEqual(response.status_code, 200, response.text)

        image_url = f"/api/admin/food-images/{self.ids['image']}/delete"
        self.assertEqual((await self.client.post(image_url, headers=owner)).status_code, 403)
        self.assertEqual((await self.client.post(image_url, headers=admin)).status_code, 200)


    async def test_weekly_food_usage_uses_actual_quantities_and_keeps_units_separate(self):
        owner = self.headers["route-owner"]
        other = self.headers["route-other"]
        start = get_current_week_start()
        async with self.sessions() as db:
            pantry = await db.get(PantryItem, self.ids["pantry"])
            pantry.name = "黑虎虾仁"
            pantry.initial_weight = 400
            pantry.remaining_weight = 400
            pantry.expire_at = datetime.now() + timedelta(days=1)
            other_unit = PantryItem(
                user_id=pantry.user_id, name=pantry.name, location=pantry.location,
                initial_weight=1, remaining_weight=1, unit="斤", expire_at=pantry.expire_at,
            )
            db.add(other_unit)
            await db.flush()
            other_unit_id = other_unit.id
            db.add_all([
                UserCarbonLog(user_id=pantry.user_id, item_name=pantry.name, weight_grams=2,
                              carbon_saved_grams=0, created_at=start),
                UserCarbonLog(user_id=pantry.user_id, item_name="上周食材", weight_grams=42,
                              carbon_saved_grams=0, created_at=start - timedelta(seconds=1)),
                UserCarbonLog(user_id=pantry.user_id, item_name="下周食材", weight_grams=42,
                              carbon_saved_grams=0, created_at=start + timedelta(days=7)),
            ])
            await db.commit()

        item_id = self.ids["pantry"]
        single_url = f"/api/pantry/items/{item_id}/deduct"
        batch_url = "/api/pantry/items/batch-deduct"
        with patch("backend.routers.pantry.ai_estimate_carbon_single", return_value=(0, "", "non_vegetarian")):
            response = await self.client.post(single_url, headers=owner, json={
                "deduct_weight": 150, "recipe_name": "蒜蓉虾仁",
            })
            self.assertEqual(response.status_code, 200, response.text)

        batch = {
            "recipe_name": "虾仁滑蛋",
            "items": [
                {"item_id": item_id, "deduct_weight": 100},
                {"item_id": other_unit_id, "deduct_weight": 0.5},
            ],
        }
        with patch("backend.routers.pantry.ai_estimate_carbon_batch", return_value=(0, {}, {}, "")):
            response = await self.client.post(batch_url, headers=owner, json=batch)
            self.assertEqual(response.status_code, 200, response.text)
        duplicate = await self.client.post(batch_url, headers=owner, json={
            "items": [batch["items"][0], batch["items"][0]],
        })
        self.assertEqual(duplicate.status_code, 400)

        url = "/api/carbon/my-food-week"
        self.assertEqual((await self.client.get(url)).status_code, 401)
        self.assertEqual((await self.client.get(url, headers=other)).json()["foods"], [])
        response = await self.client.get(url, headers=owner)
        self.assertEqual(response.status_code, 200, response.text)
        data = response.json()
        self.assertEqual(data["week_start"], start.date().isoformat())
        by_unit = {food["unit"]: food for food in data["foods"]}
        self.assertEqual({unit: float(food["total_amount"]) for unit, food in by_unit.items()},
                         {"克": 250, "斤": 0.5, None: 2})
        self.assertEqual({row["recipe"]: row["amount"] for row in by_unit["克"]["entries"]},
                         {"蒜蓉虾仁": 150, "虾仁滑蛋": 100})
        async with self.sessions() as db:
            original_batch = await db.get(PantryItem, item_id)
            self.assertEqual(original_batch.remaining_weight, 150)


if __name__ == "__main__":
    unittest.main()
