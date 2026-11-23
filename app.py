import os, re, json
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.utils
from flask import Flask, render_template, request, jsonify, session
from typing import Optional
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)
app.secret_key = os.getenv("SECRET_KEY", "bk-analytics-secret-2024")

# ── Dataset paths ────────────────────────────────────────────────────────────
BASE = os.path.dirname(__file__)
DATASET_DIR = os.path.join(BASE, "dataset")

REVIEWS_CSV = os.path.join(DATASET_DIR, "burger_king_dummy_reviews_2000.csv")
MENU_CSV    = os.path.join(DATASET_DIR, "Burger king.xlsx - Table 1.csv")
ORDER_CSV   = os.path.join(DATASET_DIR, "burger_king_dummy_orders_10000.csv")

# ── Load data ────────────────────────────────────────────────────────────────
reviews_df = pd.read_csv(REVIEWS_CSV, parse_dates=["review_date"])

menu_df = pd.read_csv(MENU_CSV)
menu_df.columns = [
    "Product","Portion_Size_gm","Energy_Kcal","Carbohydrate_gm",
    "Sugar_gm","Fat_gm","Saturated_Fat_gm","Trans_Fatty_Acids_gm",
    "Protein_gm","Sodium_Content_gm"
]
numeric_menu_cols = menu_df.columns.drop("Product")
for col in numeric_menu_cols:
    menu_df[col] = pd.to_numeric(menu_df[col], errors="coerce")

orders_df = pd.read_csv(ORDER_CSV, parse_dates=["order_datetime"]) if os.path.exists(ORDER_CSV) else pd.DataFrame()

MODEL = "gemini-3.5-flash-lite"

# ── Lazy embeddings ───────────────────────────────────────────────────────────
_embedder = _emb = None

def _ensure_embeddings():
    global _embedder, _emb
    if _emb is None:
        from sentence_transformers import SentenceTransformer
        _embedder = SentenceTransformer("all-MiniLM-L6-v2")
        _emb = _embedder.encode(reviews_df["review_text"].tolist(),
                                normalize_embeddings=True, show_progress_bar=False)

# ══════════════════════════════════════════════════════════════════════════════
# TOOL FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════
def _filter_reviews(product=None, location=None, order_type=None, sentiment=None,
                    min_rating=None, max_rating=None, date_from=None, date_to=None):
    out = reviews_df
    for col, val in [("product_reviewed", product), ("location", location),
                     ("order_type", order_type), ("sentiment", sentiment)]:
        if val:
            out = out[out[col].str.lower() == val.lower()]
    if min_rating: out = out[out["rating"] >= int(min_rating)]
    if max_rating: out = out[out["rating"] <= int(max_rating)]
    if date_from:  out = out[out["review_date"] >= pd.to_datetime(date_from)]
    if date_to:    out = out[out["review_date"] <= pd.to_datetime(date_to)]
    return out

def _summarize(g):
    return {
        "n_reviews":     int(len(g)),
        "avg_rating":    round(float(g["rating"].mean()), 2),
        "sentiment_pct": (g["sentiment"].value_counts(normalize=True)*100).round(1).to_dict(),
    }

def get_stats(product:Optional[str]=None, location:Optional[str]=None,
              order_type:Optional[str]=None, sentiment:Optional[str]=None,
              min_rating:Optional[int]=None, max_rating:Optional[int]=None,
              date_from:Optional[str]=None, date_to:Optional[str]=None,
              group_by:Optional[str]=None) -> dict:
    """Exact review counts, average rating and sentiment split.
    Args:
        product: Product name e.g. 'Chicken Whopper'.
        location: Outlet/city e.g. 'Pune'.
        order_type: One of Dine-in, Delivery, Takeaway.
        sentiment: One of Positive, Neutral, Negative.
        min_rating: Minimum star rating (1-5).
        max_rating: Maximum star rating (1-5).
        date_from: Start date YYYY-MM-DD.
        date_to: End date YYYY-MM-DD.
        group_by: One of product, location, order_type, sentiment, month.
    """
    sub = _filter_reviews(product, location, order_type, sentiment, min_rating, max_rating, date_from, date_to)
    if not len(sub): return {"n_reviews": 0, "note": "No reviews match."}
    cols = {"product":"product_reviewed","location":"location","order_type":"order_type","sentiment":"sentiment"}
    if group_by in cols:
        return {"groups": {str(k): _summarize(g) for k,g in sub.groupby(cols[group_by])}}
    if group_by == "month":
        return {"groups": {str(k): _summarize(g) for k,g in sub.groupby(sub["review_date"].dt.to_period("M"))}}
    return _summarize(sub)

def search_reviews(query:str, product:Optional[str]=None, location:Optional[str]=None,
                   order_type:Optional[str]=None, sentiment:Optional[str]=None,
                   min_rating:Optional[int]=None, max_rating:Optional[int]=None, k:int=8) -> list:
    """Semantic search over review text.
    Args:
        query: Natural-language search query.
        product: Optional product filter.
        location: Optional location filter.
        order_type: Optional order-type filter.
        sentiment: Optional sentiment filter.
        min_rating: Optional min star rating.
        max_rating: Optional max star rating.
        k: Number of results (max 15).
    """
    _ensure_embeddings()
    sub = _filter_reviews(product, location, order_type, sentiment, min_rating, max_rating)
    if not len(sub): return []
    idx = sub.index.to_numpy()
    q   = _embedder.encode([query], normalize_embeddings=True)[0]
    top = idx[np.argsort(-(_emb[idx] @ q))[:min(int(k),15)]]
    rows = reviews_df.loc[top, ["review_id","review_date","location","product_reviewed","order_type","rating","review_text"]].copy()
    rows["review_date"] = rows["review_date"].dt.strftime("%Y-%m-%d")
    return rows.to_dict("records")

def get_menu_stats(metric:Optional[str]=None, order:str="highest", top_n:int=5) -> dict:
    """Nutritional stats and product rankings.
    Args:
        metric: calories, protein, fat, sugar, sodium, carbohydrates, portion size.
        order: highest or lowest.
        top_n: Number of products to list.
    """
    mm = {"calories":"Energy_Kcal","energy":"Energy_Kcal","protein":"Protein_gm","fat":"Fat_gm",
          "sugar":"Sugar_gm","sodium":"Sodium_Content_gm","carbohydrates":"Carbohydrate_gm",
          "carbs":"Carbohydrate_gm","portion size":"Portion_Size_gm"}
    if metric is None:
        return {"total_products":int(len(menu_df)),
                "average_nutrition":menu_df[numeric_menu_cols].mean().round(2).to_dict(),
                "highest_calorie_product":menu_df.loc[menu_df["Energy_Kcal"].idxmax(),"Product"]}
    col = mm.get(metric.lower())
    if not col: return {"error":"Metric not recognized."}
    res = menu_df.sort_values(col, ascending=(order.lower()=="lowest")).head(max(1,min(int(top_n),20)))
    return {"metric":metric,"order":order,
            "results":res[["Product",col]].to_dict("records"),
            "average":round(float(menu_df[col].mean()),2),
            "minimum":round(float(menu_df[col].min()),2),
            "maximum":round(float(menu_df[col].max()),2)}

def search_menu(query:str, k:int=8) -> list:
    """Search menu products by name keyword.
    Args:
        query: Product name or keywords.
        k: Number of results (max 15).
    """
    words = [w for w in re.findall(r"[a-z0-9]+", query.lower()) if len(w)>2]
    if not words: return []
    scores = menu_df["Product"].str.lower().apply(lambda n: sum(w in n for w in words))
    matches = menu_df.loc[scores>0].copy()
    matches["_s"] = scores[scores>0]
    return matches.sort_values("_s",ascending=False).head(max(1,min(int(k),15))).drop(columns="_s").to_dict("records")

def _filter_orders(city=None, restaurant_id=None, order_type=None,
                   payment_status=None, order_status=None, date_from=None, date_to=None):
    if orders_df.empty: return orders_df
    out = orders_df
    if city:           out = out[out["city"].str.lower()==city.lower()]
    if restaurant_id:  out = out[out["restaurant_id"].astype(str)==str(restaurant_id)]
    if order_type:     out = out[out["order_type"].str.lower()==order_type.lower()]
    if payment_status: out = out[out["payment_status"].str.lower()==payment_status.lower()]
    if order_status:   out = out[out["order_status"].str.lower()==order_status.lower()]
    if date_from:      out = out[out["order_datetime"]>=pd.to_datetime(date_from)]
    if date_to:        out = out[out["order_datetime"]<=pd.to_datetime(date_to)]
    return out

def get_order_stats(city:Optional[str]=None, restaurant_id:Optional[str]=None,
                    order_type:Optional[str]=None, payment_status:Optional[str]=None,
                    order_status:Optional[str]=None, date_from:Optional[str]=None,
                    date_to:Optional[str]=None, group_by:Optional[str]=None) -> dict:
    """Order counts, revenue, delivery time, ratings.
    Args:
        city: City name e.g. 'Mumbai'.
        restaurant_id: Restaurant ID string.
        order_type: Dine-in, Delivery, or Takeaway.
        payment_status: Paid, Pending, Failed.
        order_status: Completed, Cancelled, etc.
        date_from: Start date YYYY-MM-DD.
        date_to: End date YYYY-MM-DD.
        group_by: city, restaurant_id, order_type, payment_status, order_status, or month.
    """
    sub = _filter_orders(city, restaurant_id, order_type, payment_status, order_status, date_from, date_to)
    if sub.empty: return {"n_orders":0,"note":"No orders match."}
    def _os(g):
        return {"n_orders":int(len(g)),
                "total_revenue_inr":round(float(g["total_amount_inr"].sum()),2),
                "avg_rating":round(float(g["customer_rating"].mean()),2),
                "avg_delivery_min":round(float(g["delivery_time_minutes"].mean()),2)}
    if group_by=="month":
        s=sub.copy(); s["month"]=s["order_datetime"].dt.to_period("M").astype(str)
        return {"groups":{k:_os(g) for k,g in s.groupby("month")}}
    if group_by and group_by in sub.columns:
        return {"groups":{str(k):_os(g) for k,g in sub.groupby(group_by)}}
    return _os(sub)

def search_orders(query:str, k:int=8) -> list:
    """Search orders by item name or city keyword.
    Args:
        query: Product name or city keyword.
        k: Number of results (max 20).
    """
    if orders_df.empty: return []
    q = query.lower()
    mask = (orders_df["item_name"].str.lower().str.contains(q,na=False) |
            orders_df["city"].str.lower().str.contains(q,na=False))
    rows = orders_df[mask].head(min(int(k),20)).copy()
    rows["order_datetime"] = rows["order_datetime"].dt.strftime("%Y-%m-%d %H:%M:%S")
    rows = rows.astype(object).where(rows.notna(), None)
    return rows.to_dict("records")

# ══════════════════════════════════════════════════════════════════════════════
# GEMINI CLIENT & SESSION MANAGEMENT
# ══════════════════════════════════════════════════════════════════════════════
_gemini_client = None
_sessions: dict = {}   # key → google-genai chat session (server-side, persistent per key)

CHAT_CONFIGS = {
    "review": {
        "system": (
            f"You are an expert analyst for a Burger King customer-review dataset "
            f"({len(reviews_df)} reviews, from {reviews_df['review_date'].min().date()} "
            f"to {reviews_df['review_date'].max().date()}). "
            f"Products: {', '.join(sorted(reviews_df['product_reviewed'].unique()))}. "
            f"Locations: {', '.join(sorted(reviews_df['location'].unique()))}. "
            "Rules: Answer ONLY from tool results. For numbers call get_stats. "
            "For qualitative questions call search_reviews and summarise themes with review IDs. "
            "Be concise, friendly, and use markdown formatting (bullets, bold, tables where helpful). "
            "Note this is synthetic dummy data."
        ),
        "tools": [get_stats, search_reviews],
        "suggestions": [
            "What is the overall average rating?",
            "Which product has the best sentiment?",
            "What do people complain about most?",
            "Show ratings by location",
            "Top 5 most reviewed products"
        ]
    },
    "menu": {
        "system": (
            f"You are a Burger King nutrition expert for {len(menu_df)} menu products. "
            "Rules: Use get_menu_stats for numbers/rankings. Use search_menu for product lookups. "
            "Answer ONLY from tool results. Never invent nutrition values. "
            "Use markdown formatting — tables are great for comparisons. Mention units (kcal, g)."
        ),
        "tools": [get_menu_stats, search_menu],
        "suggestions": [
            "Which item has the most calories?",
            "Top 5 highest protein items",
            "Show me the lowest fat options",
            "What is the average sodium in burgers?",
            "Compare Whopper nutrition details"
        ]
    },
    "order": {
        "system": (
            f"You are a Burger King order analytics assistant for "
            f"{format(len(orders_df), ',') if not orders_df.empty else 'an'} orders dataset. "
            f"{'Cities: ' + ', '.join(sorted(orders_df['city'].unique())) + '.' if not orders_df.empty else ''} "
            "Rules: Use get_order_stats for numbers. Use search_orders for product/city lookups. "
            "Answer ONLY from tool results. Use markdown — bullets and tables preferred."
        ),
        "tools": [get_order_stats, search_orders],
        "suggestions": [
            "What is the total revenue?",
            "Which city has the most orders?",
            "Show revenue breakdown by order type",
            "Average delivery time by city",
            "Most popular items by quantity"
        ]
    }
}

def _get_gemini_client():
    global _gemini_client
    if _gemini_client is None:
        api_key = os.getenv("GEMINI_API_KEY", "").strip()
        if not api_key or api_key == "YOUR_GEMINI_API_KEY_HERE":
            return None
        from google import genai
        _gemini_client = genai.Client(api_key=api_key)
    return _gemini_client

def _get_session(key: str):
    if key in _sessions:
        return _sessions[key]
    client = _get_gemini_client()
    if not client:
        return None
    from google.genai import types
    cfg = CHAT_CONFIGS[key]
    _sessions[key] = client.chats.create(
        model=MODEL,
        config=types.GenerateContentConfig(
            system_instruction=cfg["system"],
            tools=cfg["tools"],
            temperature=0.2,
        ),
    )
    return _sessions[key]

# ══════════════════════════════════════════════════════════════════════════════
# DASHBOARD HELPERS
# ══════════════════════════════════════════════════════════════════════════════
PLOTLY_THEME = "plotly_white"

def _chart(fig):
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Segoe UI, system-ui, sans-serif", size=13),
        margin=dict(l=20, r=20, t=50, b=30),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    return json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)

# ══════════════════════════════════════════════════════════════════════════════
# ROUTES
# ══════════════════════════════════════════════════════════════════════════════
@app.route("/")
def index():
    stats = {
        "reviews":  f"{len(reviews_df):,}",
        "products": f"{reviews_df['product_reviewed'].nunique():,}",
        "locations": f"{reviews_df['location'].nunique():,}",
        "orders":   f"{len(orders_df):,}" if not orders_df.empty else "—",
        "avg_rating": f"{reviews_df['rating'].mean():.2f}",
        "positive_pct": f"{(reviews_df['sentiment']=='Positive').mean()*100:.0f}%",
    }
    return render_template("index.html", stats=stats)

@app.route("/dashboard")
def dashboard():
    charts = []

    # 1. Sentiment pie
    sent = reviews_df["sentiment"].value_counts()
    fig1 = px.pie(values=sent.values, names=sent.index,
                  title="Overall Sentiment Breakdown",
                  color_discrete_map={"Positive":"#22c55e","Neutral":"#f59e0b","Negative":"#ef4444"})
    fig1.update_traces(textposition="inside", textinfo="percent+label", hole=0.4)
    charts.append(_chart(fig1))

    # 2. Rating distribution
    fig2 = px.histogram(reviews_df, x="rating", color="sentiment", barmode="overlay",
                        title="Rating Distribution by Sentiment",
                        color_discrete_map={"Positive":"#22c55e","Neutral":"#f59e0b","Negative":"#ef4444"},
                        category_orders={"rating":[1,2,3,4,5]})
    charts.append(_chart(fig2))

    # 3. Reviews per month
    df_m = reviews_df.copy()
    df_m["month"] = df_m["review_date"].dt.to_period("M").astype(str)
    per_month = df_m.groupby("month").size().reset_index(name="Reviews")
    fig3 = px.line(per_month, x="month", y="Reviews", title="Reviews Per Month",
                   markers=True, line_shape="spline",
                   color_discrete_sequence=["#f97316"])
    fig3.update_traces(fill="tozeroy", fillcolor="rgba(249,115,22,0.1)")
    charts.append(_chart(fig3))

    # 4. Top 10 most-reviewed products
    top_prods = reviews_df["product_reviewed"].value_counts().head(10).reset_index()
    top_prods.columns = ["Product","Reviews"]
    fig4 = px.bar(top_prods.sort_values("Reviews"), x="Reviews", y="Product",
                  orientation="h", title="Top 10 Most-Reviewed Products",
                  color="Reviews", color_continuous_scale="Oranges")
    charts.append(_chart(fig4))

    # 5. Avg rating by location
    avg_loc = reviews_df.groupby("location")["rating"].mean().round(2).reset_index()
    avg_loc.columns = ["Location","Avg Rating"]
    avg_loc = avg_loc.sort_values("Avg Rating", ascending=False)
    fig5 = px.bar(avg_loc, x="Location", y="Avg Rating",
                  title="Average Rating by Location",
                  color="Avg Rating", color_continuous_scale="RdYlGn",
                  range_color=[1,5])
    fig5.add_hline(y=reviews_df["rating"].mean(), line_dash="dash",
                   line_color="#6b7280", annotation_text="Overall avg")
    charts.append(_chart(fig5))

    # 6. Order type split (reviews)
    ot = reviews_df["order_type"].value_counts().reset_index()
    ot.columns = ["Order Type","Count"]
    fig6 = px.bar(ot, x="Order Type", y="Count", title="Reviews by Order Type",
                  color="Order Type",
                  color_discrete_sequence=["#f97316","#fb923c","#fdba74"])
    charts.append(_chart(fig6))

    # 7–8. Order-data charts (if available)
    if not orders_df.empty:
        rev_m = orders_df.copy()
        rev_m["month"] = rev_m["order_datetime"].dt.to_period("M").astype(str)
        rev_sum = rev_m.groupby("month")["total_amount_inr"].sum().reset_index(name="Revenue (₹)")
        fig7 = px.area(rev_sum, x="month", y="Revenue (₹)",
                       title="Monthly Order Revenue (₹)",
                       color_discrete_sequence=["#f97316"])
        fig7.update_traces(fillcolor="rgba(249,115,22,0.15)")
        charts.append(_chart(fig7))

        top_items = (orders_df.groupby("item_name")["quantity"].sum()
                     .sort_values(ascending=False).head(10).reset_index(name="Qty Sold"))
        fig8 = px.bar(top_items.sort_values("Qty Sold"), x="Qty Sold", y="item_name",
                      orientation="h", title="Top 10 Best-Selling Items",
                      color="Qty Sold", color_continuous_scale="Oranges")
        charts.append(_chart(fig8))

    return render_template("dashboard.html", charts=charts)

# ── Chat API (AJAX POST) ──────────────────────────────────────────────────────
@app.route("/api/chat/<page>", methods=["POST"])
def api_chat(page):
    if page not in ("review","menu","order"):
        return jsonify({"error":"Unknown chat page"}), 400
    data = request.get_json(silent=True) or {}
    msg  = (data.get("msg") or "").strip()
    if not msg:
        return jsonify({"error":"Empty message"}), 400
    chat_session = _get_session(page)
    if chat_session is None:
        return jsonify({"error":"⚠️ GEMINI_API_KEY is not set. Please add your key to the .env file and restart the server."}), 500
    try:
        response = chat_session.send_message(msg)
        return jsonify({"answer": response.text})
    except Exception as exc:
        err = str(exc)
        # Friendly messages for common errors
        if "API_KEY" in err.upper() or "401" in err:
            err = "Invalid Gemini API key. Please check your .env file."
        elif "QUOTA" in err.upper() or "429" in err:
            err = "API quota exceeded. Please wait a moment and try again."
        elif "model" in err.lower():
            err = f"Model error: {err}"
        return jsonify({"error": f"⚠️ {err}"}), 500

# ── Chat pages ────────────────────────────────────────────────────────────────
@app.route("/chat/review")
def chat_review():
    return render_template("chat_review.html",
        page="review", title="Review Chat", icon="⭐",
        subtitle="Ask anything about 2,000 Burger King customer reviews",
        suggestions=CHAT_CONFIGS["review"]["suggestions"],
        accent="#f97316")

@app.route("/chat/menu")
def chat_menu():
    return render_template("chat_menu.html",
        page="menu", title="Menu Chat", icon="🍔",
        subtitle="Explore nutrition data — calories, protein, fat & more",
        suggestions=CHAT_CONFIGS["menu"]["suggestions"],
        accent="#22c55e")

@app.route("/chat/order")
def chat_order():
    return render_template("chat_order.html",
        page="order", title="Order Chat", icon="🛒",
        subtitle="Analyse order revenue, delivery times & customer ratings",
        suggestions=CHAT_CONFIGS["order"]["suggestions"],
        accent="#3b82f6")

@app.route("/api/chat/<page>/reset", methods=["POST"])
def reset_chat(page):
    if page in _sessions:
        del _sessions[page]
    return jsonify({"ok": True})

if __name__ == "__main__":
    api_key = os.getenv("GEMINI_API_KEY","")
    if not api_key or api_key == "YOUR_GEMINI_API_KEY_HERE":
        print("\n⚠️  WARNING: GEMINI_API_KEY not set in .env — chat will return an error.\n")
    app.run(host="0.0.0.0", port=5000, debug=True)
