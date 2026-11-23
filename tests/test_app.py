"""
tests/test_app.py
-----------------
Runs WITHOUT a real Gemini API key – the /api/chat/* endpoint is tested with
a mock that returns a canned answer, so the suite works offline.

Run:
    cd /home/apk/Project/IDA
    venv/bin/pytest tests/ -v
"""

import json
import os
import sys
import types as builtin_types
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

# ── make sure the project root is on sys.path ─────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

# ── Patch google.genai BEFORE importing app ───────────────────────────────────
# We build a minimal fake module so app.py never tries to reach the network.
fake_genai      = MagicMock()
fake_types      = MagicMock()
fake_chat_resp  = MagicMock()
fake_chat_resp.text = "Mocked Gemini answer"
fake_session    = MagicMock()
fake_session.send_message.return_value = fake_chat_resp
fake_client     = MagicMock()
fake_client.chats.create.return_value = fake_session
fake_genai.Client.return_value = fake_client

sys.modules.setdefault("google",           MagicMock())
sys.modules.setdefault("google.genai",     fake_genai)
sys.modules.setdefault("google.genai.types", fake_types)

# Patch sentence_transformers so the import doesn't fail when not installed
fake_st = MagicMock()
sys.modules.setdefault("sentence_transformers", fake_st)

# Now import the Flask app
os.environ.setdefault("GEMINI_API_KEY", "test-key-not-real")
import app as bk_app  # noqa: E402  (import after sys.path and mocks are ready)


@pytest.fixture()
def client():
    bk_app.app.config["TESTING"] = True
    with bk_app.app.test_client() as c:
        yield c


# ═══════════════════════════════════════════════════════════════════════════════
# 1.  Routes – basic HTTP status checks
# ═══════════════════════════════════════════════════════════════════════════════
class TestRoutes:
    def test_home_200(self, client):
        r = client.get("/")
        assert r.status_code == 200
        assert b"BK Analytics" in r.data or b"Burger King" in r.data

    def test_dashboard_200(self, client):
        r = client.get("/dashboard")
        assert r.status_code == 200
        # Plotly JSON should be embedded in the page
        assert b"Plotly" in r.data or b"chart" in r.data

    def test_review_chat_page_200(self, client):
        assert client.get("/chat/review").status_code == 200

    def test_menu_chat_page_200(self, client):
        assert client.get("/chat/menu").status_code == 200

    def test_order_chat_page_200(self, client):
        assert client.get("/chat/order").status_code == 200

    def test_unknown_route_404(self, client):
        assert client.get("/nonexistent").status_code == 404


# ═══════════════════════════════════════════════════════════════════════════════
# 2.  Chat API – JSON endpoint
# ═══════════════════════════════════════════════════════════════════════════════
class TestChatAPI:
    def _post(self, client, page, msg):
        return client.post(
            f"/api/chat/{page}",
            data=json.dumps({"msg": msg}),
            content_type="application/json",
        )

    def test_review_chat_returns_answer(self, client):
        # Inject fake session so no real API call is made
        bk_app._sessions["review"] = fake_session
        r = self._post(client, "review", "What is the average rating?")
        assert r.status_code == 200
        data = json.loads(r.data)
        assert "answer" in data
        assert data["answer"] == "Mocked Gemini answer"

    def test_menu_chat_returns_answer(self, client):
        bk_app._sessions["menu"] = fake_session
        r = self._post(client, "menu", "What has the most calories?")
        assert r.status_code == 200
        assert "answer" in json.loads(r.data)

    def test_order_chat_returns_answer(self, client):
        bk_app._sessions["order"] = fake_session
        r = self._post(client, "order", "Total revenue last month?")
        assert r.status_code == 200
        assert "answer" in json.loads(r.data)

    def test_empty_message_returns_400(self, client):
        r = self._post(client, "review", "")
        assert r.status_code == 400

    def test_unknown_page_returns_400(self, client):
        r = self._post(client, "unknown", "hello")
        assert r.status_code == 400


# ═══════════════════════════════════════════════════════════════════════════════
# 3.  Review tools (pure-Python, no Gemini)
# ═══════════════════════════════════════════════════════════════════════════════
class TestReviewTools:
    def test_get_stats_overall(self):
        result = bk_app.get_stats()
        assert "n_reviews" in result
        assert result["n_reviews"] == len(bk_app.reviews_df)
        assert "avg_rating" in result
        assert 1.0 <= result["avg_rating"] <= 5.0

    def test_get_stats_sentiment_filter(self):
        result = bk_app.get_stats(sentiment="Positive")
        pos_count = int((bk_app.reviews_df["sentiment"] == "Positive").sum())
        assert result["n_reviews"] == pos_count

    def test_get_stats_group_by_sentiment(self):
        result = bk_app.get_stats(group_by="sentiment")
        assert "groups" in result
        assert len(result["groups"]) > 0

    def test_get_stats_no_match(self):
        result = bk_app.get_stats(location="NONEXISTENT_LOCATION_XYZ")
        assert result["n_reviews"] == 0

    def test_get_stats_rating_range(self):
        result = bk_app.get_stats(min_rating=4, max_rating=5)
        assert result["n_reviews"] > 0
        # Confirm all returned rows have rating >= 4
        sub = bk_app.reviews_df[bk_app.reviews_df["rating"] >= 4]
        assert result["n_reviews"] == len(sub)


# ═══════════════════════════════════════════════════════════════════════════════
# 4.  Menu tools
# ═══════════════════════════════════════════════════════════════════════════════
class TestMenuTools:
    def test_get_menu_stats_no_metric(self):
        result = bk_app.get_menu_stats()
        assert "total_products" in result
        assert result["total_products"] == len(bk_app.menu_df)

    def test_get_menu_stats_calories_highest(self):
        result = bk_app.get_menu_stats(metric="calories", order="highest", top_n=3)
        assert "results" in result
        assert len(result["results"]) <= 3
        # First result should have highest Energy_Kcal
        energies = [r["Energy_Kcal"] for r in result["results"]]
        assert energies == sorted(energies, reverse=True)

    def test_get_menu_stats_protein_lowest(self):
        result = bk_app.get_menu_stats(metric="protein", order="lowest", top_n=5)
        proteins = [r["Protein_gm"] for r in result["results"]]
        assert proteins == sorted(proteins)

    def test_get_menu_stats_bad_metric(self):
        result = bk_app.get_menu_stats(metric="BADMETRIC")
        assert "error" in result

    def test_search_menu_finds_product(self):
        # Pick a word from an actual product name
        word = bk_app.menu_df["Product"].iloc[0].split()[0].lower()
        results = bk_app.search_menu(word)
        assert isinstance(results, list)
        assert len(results) >= 1

    def test_search_menu_empty_query(self):
        results = bk_app.search_menu("")
        assert results == []


# ═══════════════════════════════════════════════════════════════════════════════
# 5.  Order tools (uses a small synthetic DataFrame if order_data.csv absent)
# ═══════════════════════════════════════════════════════════════════════════════
class TestOrderTools:
    @pytest.fixture(autouse=True)
    def inject_fake_orders(self):
        """Replace orders_df with a small synthetic one so tests are deterministic."""
        fake = pd.DataFrame({
            "order_id":              [1, 2, 3, 4, 5],
            "customer_id":           ["C1","C2","C1","C3","C2"],
            "customer_name":         ["Alice","Bob","Alice","Carol","Bob"],
            "customer_age":          [25, 30, 25, 22, 30],
            "customer_gender":       ["F","M","F","F","M"],
            "city":                  ["Mumbai","Delhi","Mumbai","Pune","Delhi"],
            "restaurant_id":         ["R1","R2","R1","R3","R2"],
            "order_datetime":        pd.to_datetime([
                "2024-01-05","2024-01-10","2024-02-01","2024-02-15","2024-03-01"]),
            "item_name":             ["Whopper","Fries","Whopper","Zinger","Fries"],
            "item_category":         ["Burger","Sides","Burger","Burger","Sides"],
            "quantity":              [1, 2, 1, 1, 3],
            "unit_price_inr":        [150, 60, 150, 130, 60],
            "subtotal_inr":          [150,120,150,130,180],
            "discount_percent":      [0, 5, 0, 10, 5],
            "discount_amount_inr":   [0, 6, 0, 13, 9],
            "tax_inr":               [15, 11, 15, 12, 17],
            "delivery_fee_inr":      [30, 30, 30, 30, 30],
            "total_amount_inr":      [195, 155, 195, 159, 218],
            "order_type":            ["Delivery","Dine-in","Delivery","Takeaway","Dine-in"],
            "payment_method":        ["Card","Cash","UPI","Card","Cash"],
            "payment_status":        ["Paid","Paid","Paid","Paid","Paid"],
            "order_status":          ["Completed","Completed","Completed","Completed","Completed"],
            "delivery_time_minutes": [30, 0, 35, 20, 0],
            "customer_rating":       [4, 5, 3, 4, 5],
            "coupon_used":           [False,True,False,True,False],
        })
        original = bk_app.orders_df
        bk_app.orders_df = fake
        yield
        bk_app.orders_df = original

    def test_get_order_stats_overall(self):
        result = bk_app.get_order_stats()
        assert result["n_orders"] == 5
        assert result["total_revenue_inr"] == pytest.approx(195+155+195+159+218)

    def test_get_order_stats_city_filter(self):
        result = bk_app.get_order_stats(city="Mumbai")
        assert result["n_orders"] == 2

    def test_get_order_stats_group_by_city(self):
        result = bk_app.get_order_stats(group_by="city")
        assert "groups" in result
        assert "Mumbai" in result["groups"]
        assert result["groups"]["Mumbai"]["n_orders"] == 2

    def test_get_order_stats_no_match(self):
        result = bk_app.get_order_stats(city="NOWHERE")
        assert result["n_orders"] == 0

    def test_search_orders_by_item(self):
        results = bk_app.search_orders("Whopper")
        assert len(results) == 2
        assert all("Whopper" in r["item_name"] for r in results)

    def test_search_orders_by_city(self):
        results = bk_app.search_orders("Delhi")
        assert len(results) == 2

    def test_search_orders_no_match(self):
        results = bk_app.search_orders("NOMATCH_XYZ")
        assert results == []


# ═══════════════════════════════════════════════════════════════════════════════
# 6.  Data integrity checks
# ═══════════════════════════════════════════════════════════════════════════════
class TestDataIntegrity:
    def test_reviews_has_expected_columns(self):
        required = {"review_id","review_date","location","product_reviewed",
                    "order_type","rating","review_text","sentiment"}
        assert required.issubset(bk_app.reviews_df.columns)

    def test_reviews_rating_range(self):
        assert bk_app.reviews_df["rating"].between(1, 5).all()

    def test_reviews_sentiment_values(self):
        valid = {"Positive","Neutral","Negative"}
        assert set(bk_app.reviews_df["sentiment"].unique()).issubset(valid)

    def test_menu_has_expected_columns(self):
        assert "Product"    in bk_app.menu_df.columns
        assert "Energy_Kcal" in bk_app.menu_df.columns

    def test_menu_no_negative_calories(self):
        assert (bk_app.menu_df["Energy_Kcal"].dropna() >= 0).all()
