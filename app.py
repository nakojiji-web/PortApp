"""
💼 Real-time Portfolio Tracker
Streamlit + yfinance + CoinGecko (Requests) + Plotly

รันด้วย:
    pip install "streamlit>=1.37" yfinance requests plotly pandas numpy
    streamlit run app.py
"""

from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import requests
import streamlit as st
import yfinance as yf

# ─────────────────────────────── Config ───────────────────────────────
st.set_page_config(
    page_title="Portfolio Tracker",
    page_icon="💼",
    layout="wide",
    initial_sidebar_state="collapsed",
)

TYPES = ["SET", "US", "Crypto", "Cash"]
COLUMNS = ["Ticker", "Type", "Quantity", "Avg Cost"]
TZ = timezone(timedelta(hours=7))  # Asia/Bangkok
PORTFOLIO_FILE = Path(__file__).with_name("portfolio.csv")
FX_SYMBOL = "THB=X"  # THB ต่อ 1 USD
DEFAULT_FX = 35.0  # ใช้เมื่อดึงอัตราแลกเปลี่ยนไม่ได้เลย
CACHE_TTL = 20  # วินาที
PALETTE = ["#6366f1", "#22c55e", "#f59e0b", "#ef4444", "#06b6d4",
           "#a855f7", "#ec4899", "#84cc16", "#14b8a6", "#f97316"]

COINGECKO_URL = "https://api.coingecko.com/api/v3/simple/price"
COINGECKO_IDS = {
    "BTC": "bitcoin", "ETH": "ethereum", "SOL": "solana", "BNB": "binancecoin",
    "XRP": "ripple", "ADA": "cardano", "DOGE": "dogecoin", "USDT": "tether",
    "USDC": "usd-coin", "DOT": "polkadot", "AVAX": "avalanche-2",
    "LINK": "chainlink", "LTC": "litecoin", "TRX": "tron", "TON": "the-open-network",
    "SUI": "sui", "MATIC": "matic-network",
}

DEFAULT_PORTFOLIO = pd.DataFrame(
    [
        ["PTT", "SET", 1000.0, 34.0],
        ["CPALL", "SET", 500.0, 58.0],
        ["AAPL", "US", 10.0, 170.0],
        ["NVDA", "US", 5.0, 120.0],
        ["BTC", "Crypto", 0.05, 60000.0],
        ["ETH", "Crypto", 0.5, 2500.0],
        ["THB", "Cash", 50000.0, 1.0],
        ["USD", "Cash", 1000.0, 1.0],
    ],
    columns=COLUMNS,
)

st.markdown(
    """
    <style>
    .block-container {padding-top: 1.6rem; padding-bottom: 3rem; max-width: 1200px;}
    div[data-testid="stMetric"] {
        background: rgba(128,128,128,.08);
        border: 1px solid rgba(128,128,128,.18);
        border-radius: 16px;
        padding: 14px 18px;
    }
    div[data-testid="stMetricValue"] {font-weight: 700;}
    div[data-testid="stMetricLabel"] {opacity: .75;}
    @media (max-width: 640px) {
        .block-container {padding-left: .8rem; padding-right: .8rem;}
        div[data-testid="stMetricValue"] {font-size: 1.45rem;}
        h1 {font-size: 1.6rem !important;}
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ─────────────────────────── Helper functions ──────────────────────────
def normalize_df(df: pd.DataFrame) -> pd.DataFrame:
    d = df.reindex(columns=COLUMNS).copy()
    d["Ticker"] = d["Ticker"].fillna("").astype(str)
    d["Type"] = d["Type"].fillna("").astype(str)
    d["Quantity"] = pd.to_numeric(d["Quantity"], errors="coerce").fillna(0.0)
    d["Avg Cost"] = pd.to_numeric(d["Avg Cost"], errors="coerce").fillna(0.0)
    return d.reset_index(drop=True)


def load_initial() -> pd.DataFrame:
    if PORTFOLIO_FILE.exists():
        try:
            raw = pd.read_csv(PORTFOLIO_FILE, dtype={"Ticker": str, "Type": str},
                              keep_default_na=False)
            return normalize_df(raw)
        except Exception:
            pass
    return DEFAULT_PORTFOLIO.copy()


def save_portfolio(df: pd.DataFrame) -> None:
    try:
        csv = df.dropna(how="all").to_csv(index=False)
        if not PORTFOLIO_FILE.exists() or PORTFOLIO_FILE.read_text(encoding="utf-8") != csv:
            PORTFOLIO_FILE.write_text(csv, encoding="utf-8")
    except Exception:
        pass  # เช่น รันบน host ที่เขียนไฟล์ไม่ได้


def clean_portfolio(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame(columns=COLUMNS)
    d = normalize_df(df)
    d["Ticker"] = d["Ticker"].str.strip().str.upper()
    d = d[(d["Ticker"] != "") & d["Type"].isin(TYPES) & (d["Quantity"] != 0)]
    bad_cash = (d["Type"] == "Cash") & ~d["Ticker"].isin(["THB", "USD"])
    return d[~bad_cash].reset_index(drop=True)


def norm_symbol(asset_type: str, ticker: str) -> str:
    s = str(ticker).strip().upper()
    if asset_type == "SET":
        return s if s.endswith(".BK") else f"{s}.BK"
    if asset_type == "Crypto":
        return s.replace("-USD", "").replace("/USD", "")
    return s


def conv_factor(src: str, dst: str, fx: float) -> float:
    """แปลงสกุลเงิน src → dst (fx = THB ต่อ 1 USD)"""
    if src == dst:
        return 1.0
    return fx if (src == "USD" and dst == "THB") else 1.0 / fx


def _valid(x) -> bool:
    try:
        return x is not None and np.isfinite(float(x)) and float(x) > 0
    except (TypeError, ValueError):
        return False


# ───────────────────────────── Data fetching ───────────────────────────
def _yf_quote(symbol: str):
    try:
        tk = yf.Ticker(symbol)
        price = prev = None
        try:
            fi = tk.fast_info
            price, prev = fi["last_price"], fi["previous_close"]
        except Exception:
            pass
        if not _valid(price):
            hist = tk.history(period="5d", auto_adjust=False)["Close"].dropna()
            if len(hist):
                price = float(hist.iloc[-1])
                prev = float(hist.iloc[-2]) if len(hist) > 1 else None
        if _valid(price):
            return {"price": float(price), "prev": float(prev) if _valid(prev) else None,
                    "source": "Yahoo Finance"}
    except Exception:
        pass
    return None


def _coingecko_prices(symbols):
    ids = {s: COINGECKO_IDS[s] for s in symbols if s in COINGECKO_IDS}
    out = {}
    if not ids:
        return out
    try:
        r = requests.get(
            COINGECKO_URL,
            params={"ids": ",".join(sorted(set(ids.values()))), "vs_currencies": "usd",
                    "include_24hr_change": "true"},
            headers={"accept": "application/json"},
            timeout=10,
        )
        r.raise_for_status()
        data = r.json()
        for sym, cid in ids.items():
            d = data.get(cid)
            if d and _valid(d.get("usd")):
                p = float(d["usd"])
                chg = d.get("usd_24h_change")
                prev = p / (1 + chg / 100) if chg is not None and chg > -100 else None
                out[sym] = {"price": p, "prev": prev, "source": "CoinGecko"}
    except Exception:
        pass
    return out


@st.cache_data(ttl=CACHE_TTL, show_spinner="กำลังดึงราคาตลาดล่าสุด...")
def fetch_market_data(assets: tuple):
    """assets = ((type, symbol), ...) → quotes + เวลาที่ดึง (cache สั้น ๆ; ปุ่ม Refresh จะล้าง cache)"""
    quotes = {}
    cg = _coingecko_prices([s for t, s in assets if t == "Crypto"])
    jobs = {("FX", FX_SYMBOL): FX_SYMBOL}
    for t, s in assets:
        if t == "Crypto":
            if s in cg:
                quotes[(t, s)] = cg[s]
            else:  # fallback → yfinance (เช่น BTC-USD)
                jobs[(t, s)] = f"{s}-USD"
        elif t in ("SET", "US"):
            jobs[(t, s)] = s
    keys = list(jobs)
    with ThreadPoolExecutor(max_workers=8) as ex:
        results = list(ex.map(_yf_quote, [jobs[k] for k in keys]))
    for k, res in zip(keys, results):
        if res:
            quotes[k] = res
    return {"quotes": quotes, "fetched_at": datetime.now(TZ)}


# ───────────────────────────── Valuation ───────────────────────────────
def build_valuation(port: pd.DataFrame, quotes: dict, fx: float, ccy: str):
    rows, missing = [], []
    for rec in port.to_dict("records"):
        t, qty, avg = rec["Type"], float(rec["Quantity"]), float(rec["Avg Cost"])
        sym = norm_symbol(t, rec["Ticker"])
        if t == "Cash":
            native, price, prev, avg_disp, price_disp = sym, 1.0, None, np.nan, np.nan
            cost_unit = 1.0
            asset = f"{sym} Cash"
            group = f"Cash ({sym})"
        else:
            native = "THB" if t == "SET" else "USD"
            q = quotes.get((t, sym))
            if q:
                price, prev = q["price"], q.get("prev")
            else:
                price, prev = avg, None
                missing.append(sym)
            avg_disp, price_disp, cost_unit = avg, price, avg
            asset = sym.replace(".BK", "")
            group = {"SET": "Thai Stocks", "US": "US Stocks", "Crypto": "Crypto"}[t]
        f = conv_factor(native, ccy, fx)
        value = qty * price * f
        cost = qty * cost_unit * f
        pnl = value - cost
        rows.append({
            "asset": asset, "type": t, "group": group, "native": native, "qty": qty,
            "avg": avg_disp, "price": price_disp,
            "day": (price / prev - 1) * 100 if (_valid(prev) and t != "Cash") else np.nan,
            "value": value, "cost": cost, "pnl": pnl,
            "pnl_pct": pnl / cost * 100 if (cost > 0 and t != "Cash") else np.nan,
        })
    val = pd.DataFrame(rows).sort_values("value", ascending=False).reset_index(drop=True)
    return val, missing


def fmt_price(v: float) -> str:
    return f"{v:,.2f}" if abs(v) >= 1 else f"{v:,.4f}"


def color_pl(v):
    if pd.isna(v) or v == 0:
        return ""
    return "color: #16a34a; font-weight: 600" if v > 0 else "color: #dc2626; font-weight: 600"


# ───────────────────────────── Dashboard ───────────────────────────────
def make_dashboard(run_every):
    @st.fragment(run_every=run_every)
    def dashboard():
        ccy = st.session_state.display_ccy
        sym = "฿" if ccy == "THB" else "$"
        port = clean_portfolio(st.session_state.portfolio)
        if port.empty:
            st.info("ยังไม่มีรายการในพอร์ต — ไปที่แท็บ “จัดการพอร์ต” เพื่อเพิ่มสินทรัพย์")
            return

        assets = tuple(sorted({
            (r["Type"], norm_symbol(r["Type"], r["Ticker"]))
            for r in port.to_dict("records") if r["Type"] != "Cash"
        }))
        data = fetch_market_data(assets)
        quotes = data["quotes"]

        fx_q = quotes.get(("FX", FX_SYMBOL))
        if fx_q:
            fx = fx_q["price"]
            st.session_state.last_fx = fx
        else:
            fx = st.session_state.get("last_fx", DEFAULT_FX)
            st.warning(f"ดึงอัตราแลกเปลี่ยน USD/THB ไม่สำเร็จ — ใช้ค่าล่าสุด/ค่าประมาณ {fx:,.2f}")

        val, missing = build_valuation(port, quotes, fx, ccy)
        if missing:
            st.warning("ดึงราคาไม่สำเร็จ: " + ", ".join(missing)
                       + " — ใช้ราคาต้นทุนแทนชั่วคราว (ตรวจสอบ Ticker หรือกด Refresh)")

        # ── สรุปภาพรวม ──
        net = val["value"].sum()
        invested = val.loc[val["type"] != "Cash", "cost"].sum()
        pnl = val["pnl"].sum()
        pnl_pct = pnl / invested * 100 if invested > 0 else 0.0

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("💰 Net Worth", f"{sym}{net:,.2f}")
        m2.metric("📈 Total P/L", f"{'+' if pnl >= 0 else '-'}{sym}{abs(pnl):,.2f}",
                  f"{pnl_pct:+.2f}%")
        m3.metric("🧾 Invested Cost", f"{sym}{invested:,.2f}", help="ต้นทุนรวมของสินทรัพย์ (ไม่รวมเงินสด)")
        m4.metric("💱 USD/THB", f"{fx:,.4f}")

        sources = sorted({q["source"] for q in quotes.values()})
        st.caption(f"🕒 อัปเดตล่าสุด {data['fetched_at']:%H:%M:%S} · แหล่งข้อมูล: {', '.join(sources) or '-'}")

        # ── Asset Allocation ──
        st.subheader("🍩 Asset Allocation")
        view = st.radio("มุมมอง", ["รายสินทรัพย์", "ตามประเภท"], horizontal=True,
                        key="alloc_view", label_visibility="collapsed")
        label = "asset" if view == "รายสินทรัพย์" else "group"
        alloc = (val[val["value"] > 0].groupby(label, as_index=False)["value"].sum()
                 .sort_values("value", ascending=False).reset_index(drop=True))
        alloc["weight"] = alloc["value"] / alloc["value"].sum() * 100

        c_chart, c_tbl = st.columns([1.1, 1])
        with c_chart:
            fig = px.pie(alloc, names=label, values="value", hole=0.62,
                         color_discrete_sequence=PALETTE)
            fig.update_traces(
                sort=False, textinfo="percent", textposition="inside",
                marker=dict(line=dict(color="rgba(0,0,0,0)", width=2)),
                hovertemplate="<b>%{label}</b><br>%{value:,.2f} " + ccy + "<br>%{percent}<extra></extra>",
            )
            fig.update_layout(
                height=380, margin=dict(t=10, b=10, l=10, r=10),
                uniformtext_minsize=10, uniformtext_mode="hide",
                legend=dict(orientation="h", yanchor="top", y=-0.02, xanchor="center", x=0.5),
                annotations=[dict(text=f"<b>{alloc['value'].sum():,.0f}</b><br>{ccy}",
                                  x=0.5, y=0.5, font_size=18, showarrow=False)],
            )
            st.plotly_chart(fig)
        with c_tbl:
            show = pd.DataFrame({
                "Asset": alloc[label],
                f"Value ({ccy})": alloc["value"].round(2),
                "Weight": alloc["weight"],
            })
            st.dataframe(
                show, hide_index=True,
                column_config={"Weight": st.column_config.ProgressColumn(
                    "Weight", format="%.1f%%", min_value=0, max_value=100)},
            )

        # ── ตารางสรุปรายสินทรัพย์ ──
        st.subheader("📋 รายละเอียดสินทรัพย์")
        v_col, p_col = f"Value ({ccy})", f"P/L ({ccy})"
        disp = pd.DataFrame({
            "Asset": val["asset"], "Type": val["type"], "Ccy": val["native"],
            "Qty": val["qty"], "Avg Cost": val["avg"], "Price": val["price"],
            "Day %": val["day"], v_col: val["value"], p_col: val["pnl"],
            "P/L %": val["pnl_pct"],
        })
        styler = disp.style.format({
            "Qty": lambda v: f"{v:,.8g}",
            "Avg Cost": fmt_price, "Price": fmt_price,
            "Day %": "{:+.2f}%", "P/L %": "{:+.2f}%",
            v_col: "{:,.2f}", p_col: "{:+,.2f}",
        }, na_rep="–")
        apply_map = getattr(styler, "map", None) or styler.applymap  # pandas ≥2.1 / เก่ากว่า
        styler = apply_map(color_pl, subset=["Day %", p_col, "P/L %"])
        st.dataframe(styler, hide_index=True)

        st.caption(
            "ℹ️ ราคาหุ้นจาก Yahoo Finance อาจล่าช้าตามเงื่อนไขของตลาด · "
            "P/L ของสินทรัพย์ USD คำนวณทั้งต้นทุนและมูลค่าด้วยอัตราแลกเปลี่ยนปัจจุบัน "
            "(ไม่รวมผลจากค่าเงินย้อนหลัง) · เพื่อการติดตามเท่านั้น ไม่ใช่คำแนะนำการลงทุน"
        )

    return dashboard


# ───────────────────────────────── App ─────────────────────────────────
if "portfolio_base" not in st.session_state:
    st.session_state.portfolio_base = load_initial()
    st.session_state.portfolio = st.session_state.portfolio_base.copy()
    st.session_state.editor_ver = 0
st.session_state.setdefault("display_ccy", "THB")

st.title("💼 Portfolio Tracker")
st.caption("ติดตามมูลค่าพอร์ตแบบ Real-time: หุ้นไทย · หุ้นสหรัฐ · คริปโต · เงินสด")

# แถบควบคุม
c1, c2, c3, c4 = st.columns([1, 1.1, 1.1, 1.6])
with c1:
    if st.button("🔄 Refresh", help="ล้าง cache และดึงราคาล่าสุดทันที"):
        st.cache_data.clear()
with c2:
    auto = st.toggle("⏱️ Auto Refresh", key="auto_refresh")
with c3:
    interval = st.selectbox(
        "รอบอัปเดต", [30, 60, 120, 300], index=1, key="refresh_interval",
        format_func=lambda s: f"ทุก {s} วินาที" if s < 60 else f"ทุก {s // 60} นาที",
        disabled=not auto, label_visibility="collapsed",
    )
with c4:
    st.radio("แสดงผลเป็น", ["THB", "USD"], horizontal=True, key="display_ccy",
             format_func=lambda c: "฿ THB" if c == "THB" else "$ USD")

tab_dash, tab_edit = st.tabs(["📊 Dashboard", "🧾 จัดการพอร์ต"])

# สำคัญ: รันแท็บแก้ไขพอร์ตก่อน เพื่อให้ Dashboard ใช้ข้อมูลล่าสุดในรอบเดียวกัน
with tab_edit:
    st.markdown(
        "**วิธีกรอก:** `SET` ใส่ชื่อย่อ เช่น PTT (ระบบเติม .BK ให้) · `US` เช่น AAPL · "
        "`Crypto` เช่น BTC, ETH · `Cash` ใส่ Ticker เป็น **THB** หรือ **USD** และจำนวนเงินใน Quantity"
    )
    edited = st.data_editor(
        st.session_state.portfolio_base,
        key=f"editor_{st.session_state.editor_ver}",
        num_rows="dynamic",
        hide_index=True,
        column_config={
            "Ticker": st.column_config.TextColumn("Ticker", required=True),
            "Type": st.column_config.SelectboxColumn("Type", options=TYPES, required=True),
            "Quantity": st.column_config.NumberColumn("Quantity", min_value=0.0,
                                                       help="จำนวนหน่วยที่ถือ (Cash = จำนวนเงิน)"),
            "Avg Cost": st.column_config.NumberColumn(
                "Avg Cost", min_value=0.0,
                help="ต้นทุนเฉลี่ยต่อหน่วย ในสกุลเงินของสินทรัพย์ (SET=บาท, US/Crypto=ดอลลาร์)"),
        },
    )
    st.session_state.portfolio = edited
    save_portfolio(edited)

    chk = normalize_df(edited)
    bad = chk[(chk["Type"] == "Cash") & ~chk["Ticker"].str.strip().str.upper().isin(["THB", "USD"])
              & (chk["Ticker"].str.strip() != "")]
    if not bad.empty:
        st.warning("รายการ Cash ต้องใช้ Ticker เป็น THB หรือ USD เท่านั้น (แถวที่ไม่ถูกต้องจะถูกข้าม)")

    b1, b2 = st.columns(2)
    with b1:
        st.download_button("📤 ดาวน์โหลดพอร์ต (CSV)", edited.dropna(how="all").to_csv(index=False),
                           file_name="portfolio.csv", mime="text/csv")
    with b2:
        if st.button("↩️ รีเซ็ตเป็นข้อมูลตัวอย่าง"):
            st.session_state.portfolio_base = DEFAULT_PORTFOLIO.copy()
            st.session_state.editor_ver += 1
            st.rerun()

    up = st.file_uploader("📥 นำเข้าพอร์ตจาก CSV (คอลัมน์: Ticker, Type, Quantity, Avg Cost)", type="csv")
    if up is not None:
        sig = f"{up.name}:{up.size}"
        if st.session_state.get("last_upload") != sig:
            try:
                raw = pd.read_csv(up, dtype={"Ticker": str, "Type": str}, keep_default_na=False)
                if not set(COLUMNS).issubset(raw.columns):
                    st.error("ไฟล์ต้องมีคอลัมน์: " + ", ".join(COLUMNS))
                else:
                    st.session_state.portfolio_base = normalize_df(raw)
                    st.session_state.editor_ver += 1
                    st.session_state.last_upload = sig
                    st.rerun()
            except Exception as e:
                st.error(f"อ่านไฟล์ไม่สำเร็จ: {e}")

with tab_dash:
    make_dashboard(interval if auto else None)()
