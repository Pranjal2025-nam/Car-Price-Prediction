"""
AutoVerse – AI Car Price Predictor & Showroom
Run:  streamlit run app.py
Needs car_model.pkl, scaler.pkl, columns.pkl in the same folder.
"""
import base64
import datetime as dt
import random
import string
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import requests
import streamlit as st
import streamlit.components.v1 as components
from sklearn.preprocessing import PolynomialFeatures

st.set_page_config(page_title="AutoVerse · Car Price & Showroom", page_icon="🏎️",
                   layout="wide", initial_sidebar_state="collapsed")

BASE = Path(__file__).parent
CURRENT_YEAR = 2026
POLY_COLS = ["Car_age", "km_driven", "engine_cc", "power_bhp"]

# ───────────────────────── catalogue ─────────────────────────
# model: (brand, [wikipedia titles], engine_cc, bhp, mileage, seats, default fuel)
CARS = {
    "3 Series":      ("BMW",      ["BMW 3 Series"], 1998, 255, 14.0, 5, "Petrol"),
    "X1":            ("BMW",      ["BMW X1"], 1995, 187, 15.0, 5, "Diesel"),
    "C-Class":       ("Mercedes", ["Mercedes-Benz C-Class"], 1991, 194, 15.0, 5, "Petrol"),
    "GLA":           ("Mercedes", ["Mercedes-Benz GLA-Class", "Mercedes-Benz GLA"], 1332, 160, 17.0, 5, "Petrol"),
    "Octavia":       ("Skoda",    ["Škoda Octavia"], 1395, 148, 15.0, 5, "Petrol"),
    "Rapid":         ("Skoda",    ["Škoda Rapid"], 999, 108, 18.0, 5, "Petrol"),
    "Alto":          ("Maruti",   ["Maruti Suzuki Alto", "Suzuki Alto"], 796, 47, 22.0, 5, "Petrol"),
    "Baleno":        ("Maruti",   ["Maruti Suzuki Baleno", "Suzuki Baleno"], 1197, 88, 22.0, 5, "Petrol"),
    "Brezza":        ("Maruti",   ["Maruti Suzuki Brezza", "Suzuki Vitara Brezza"], 1462, 102, 19.0, 5, "Petrol"),
    "Ertiga":        ("Maruti",   ["Maruti Suzuki Ertiga", "Suzuki Ertiga"], 1462, 103, 20.0, 7, "Petrol"),
    "Swift":         ("Maruti",   ["Maruti Suzuki Swift", "Suzuki Swift"], 1197, 82, 22.0, 5, "Petrol"),
    "Amaze":         ("Honda",    ["Honda Amaze"], 1199, 88, 18.6, 5, "Petrol"),
    "City":          ("Honda",    ["Honda City"], 1498, 119, 17.8, 5, "Petrol"),
    "WR-V":          ("Honda",    ["Honda WR-V"], 1199, 88, 18.0, 5, "Petrol"),
    "Creta":         ("Hyundai",  ["Hyundai Creta"], 1493, 113, 16.8, 5, "Diesel"),
    "Venue":         ("Hyundai",  ["Hyundai Venue"], 1197, 82, 18.0, 5, "Petrol"),
    "Verna":         ("Hyundai",  ["Hyundai Verna"], 1497, 113, 18.0, 5, "Petrol"),
    "i10":           ("Hyundai",  ["Hyundai Grand i10", "Hyundai i10"], 1197, 82, 20.0, 5, "Petrol"),
    "i20":           ("Hyundai",  ["Hyundai i20"], 1197, 82, 20.0, 5, "Petrol"),
    "Seltos":        ("Kia",      ["Kia Seltos"], 1497, 113, 15.6, 5, "Petrol"),
    "Sonet":         ("Kia",      ["Kia Sonet"], 1197, 82, 18.0, 5, "Petrol"),
    "Scorpio":       ("Mahindra", ["Mahindra Scorpio"], 2179, 130, 15.0, 7, "Diesel"),
    "Thar":          ("Mahindra", ["Mahindra Thar"], 2184, 130, 15.0, 4, "Diesel"),
    "XUV300":        ("Mahindra", ["Mahindra XUV300"], 1197, 108, 17.0, 5, "Petrol"),
    "XUV700":        ("Mahindra", ["Mahindra XUV700"], 1999, 197, 13.6, 7, "Petrol"),
    "Harrier":       ("Tata",     ["Tata Harrier"], 1956, 168, 16.0, 5, "Diesel"),
    "Nexon":         ("Tata",     ["Tata Nexon"], 1199, 118, 17.0, 5, "Petrol"),
    "Safari":        ("Tata",     ["Tata Safari"], 1956, 168, 16.0, 7, "Diesel"),
    "Tiago":         ("Tata",     ["Tata Tiago"], 1199, 84, 20.0, 5, "Petrol"),
    "Fortuner":      ("Toyota",   ["Toyota Fortuner"], 2755, 201, 14.0, 7, "Diesel"),
    "Glanza":        ("Toyota",   ["Toyota Glanza"], 1197, 82, 22.0, 5, "Petrol"),
    "Innova Crysta": ("Toyota",   ["Toyota Innova Crysta", "Toyota Innova"], 2393, 148, 13.0, 7, "Diesel"),
    "Endeavour":     ("Ford",     ["Ford Endeavour", "Ford Everest"], 1996, 168, 14.0, 7, "Diesel"),
    "EcoSport":      ("Ford",      ["Ford EcoSport"], 1498, 98, 21.0, 5, "Petrol"),
}
# Ford brand dummy exists (brand_Ford) – make sure brand list is derived from the table
BRANDS = sorted({v[0] for v in CARS.values()})
ACCENT = {"BMW": "#4aa8ff", "Mercedes": "#c8d0dc", "Skoda": "#3ddc84", "Maruti": "#4f8cff",
          "Honda": "#ff4d5e", "Hyundai": "#5ea0ff", "Kia": "#ff6a3d", "Mahindra": "#ff3d71",
          "Tata": "#45b6ff", "Toyota": "#ff5252", "Ford": "#3d7bff"}
BRAND_EMOJI = "🚘"


# ───────────────────────── model loading ─────────────────────────
@st.cache_resource(show_spinner=False)
def load_artifacts():
    model = joblib.load(BASE / "car_model.pkl")
    scaler = joblib.load(BASE / "scaler.pkl")
    columns = joblib.load(BASE / "columns.pkl")
    poly = PolynomialFeatures(degree=2, include_bias=False).fit(pd.DataFrame(np.zeros((1, len(POLY_COLS))), columns=POLY_COLS))
    return model, scaler, columns, poly


def predict_price(inp: dict) -> float:
    """Re-creates the exact training pipeline from the notebook → price in lakh."""
    model, scaler, columns, poly = load_artifacts()
    row = {c: 0 for c in columns}
    row.update(km_driven=inp["km"], engine_cc=inp["engine"], power_bhp=inp["bhp"],
               mileage_kmpl=inp["mileage"], seats=inp["seats"], Car_age=CURRENT_YEAR - inp["year"])
    for key in (f"brand_{inp['brand']}", f"model_{inp['model']}", f"fuel_type_{inp['fuel']}"):
        if key in row:
            row[key] = 1
    if inp["transmission"] == "Manual":
        row["transmission_Manual"] = 1
    owner_key = {"Second": "owner_type_Second", "Third": "owner_type_Third",
                 "Fourth & Above": "owner_type_Fourth & Above"}.get(inp["owner"])
    if owner_key:
        row[owner_key] = 1
    X = pd.DataFrame([row])
    poly_df = pd.DataFrame(poly.transform(X[POLY_COLS]), columns=poly.get_feature_names_out(POLY_COLS))
    full = pd.concat([X.drop(columns=POLY_COLS), poly_df], axis=1)
    full = full[list(scaler.feature_names_in_)]
    return float(max(model.predict(scaler.transform(full))[0], 0.45))


# ───────────────────────── images (4K) ─────────────────────────
def car_svg(color="#7c5cff") -> str:
    svg = f"""<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 3840 2160'>
<defs><radialGradient id='g' cx='50%' cy='60%' r='70%'><stop offset='0' stop-color='{color}' stop-opacity='.55'/>
<stop offset='1' stop-color='#07070f'/></radialGradient>
<linearGradient id='b' x1='0' y1='0' x2='0' y2='1'><stop offset='0' stop-color='#fff' stop-opacity='.95'/>
<stop offset='.5' stop-color='{color}'/><stop offset='1' stop-color='#05050c'/></linearGradient></defs>
<rect width='3840' height='2160' fill='url(#g)'/>
<ellipse cx='1920' cy='1580' rx='1250' ry='90' fill='#000' opacity='.55'/>
<path d='M520 1450 Q540 1250 800 1190 L1250 1170 Q1500 760 1900 740 L2450 760 Q2780 800 2960 1160
L3260 1230 Q3380 1280 3360 1450 Z' fill='url(#b)'/>
<path d='M1330 1160 Q1560 850 1900 830 L2400 850 Q2640 900 2800 1160 Z' fill='#0b1630' opacity='.9'/>
<circle cx='1000' cy='1470' r='215' fill='#0a0a14'/><circle cx='1000' cy='1470' r='130' fill='#9aa0b8'/>
<circle cx='2820' cy='1470' r='215' fill='#0a0a14'/><circle cx='2820' cy='1470' r='130' fill='#9aa0b8'/>
<rect x='3250' y='1290' width='110' height='40' rx='20' fill='#fff7c2'/></svg>"""
    return "data:image/svg+xml;base64," + base64.b64encode(svg.encode()).decode()


@st.cache_data(show_spinner=False, ttl=60 * 60 * 24)
def fetch_car_image(titles: tuple) -> str | None:
    """Looks up the car's lead photo on Wikipedia and returns a ≤3840px-wide (4K) URL."""
    headers = {"User-Agent": "AutoVerse-Streamlit/1.0 (car showroom demo)"}
    for title in titles:
        try:
            r = requests.get(
                "https://en.wikipedia.org/w/api.php",
                params={"action": "query", "titles": title, "prop": "pageimages", "piprop": "thumbnail",
                        "pithumbsize": 3840, "redirects": 1, "format": "json"},
                headers=headers, timeout=6)
            for page in r.json().get("query", {}).get("pages", {}).values():
                url = page.get("thumbnail", {}).get("source")
                if url and not url.lower().endswith(".svg"):
                    return url
                if url and ".svg" in url.lower():   # svg thumbs are png renders – fine
                    return url
        except Exception:
            continue
    return None


# ───────────────────────── styling ─────────────────────────
def h(s: str) -> str:
    return "\n".join(l.strip() for l in s.splitlines() if l.strip())


st.markdown(h("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;800&family=Space+Grotesk:wght@500;700&display=swap');
html, body, [class*="css"], .stApp { font-family: 'Outfit', sans-serif; }
#MainMenu, footer, header[data-testid="stHeader"] { visibility: hidden; height: 0; }
.block-container { padding-top: 1.2rem; max-width: 1350px; }
.stApp {
  background: linear-gradient(125deg,#07070f,#10082a,#06121f,#150826);
  background-size: 400% 400%; animation: bgmove 22s ease infinite;
}
@keyframes bgmove { 0%{background-position:0% 50%} 50%{background-position:100% 50%} 100%{background-position:0% 50%} }
.orb { position: fixed; border-radius: 50%; filter: blur(90px); opacity:.35; z-index:0; pointer-events:none; }
.o1 { width:420px;height:420px;background:#7c5cff;top:-100px;left:-100px;animation:float1 14s ease-in-out infinite; }
.o2 { width:380px;height:380px;background:#00d4ff;bottom:-120px;right:-80px;animation:float2 17s ease-in-out infinite; }
.o3 { width:300px;height:300px;background:#ff3d9a;top:40%;left:55%;opacity:.18;animation:float1 20s ease-in-out infinite reverse; }
@keyframes float1 { 0%,100%{transform:translate(0,0)} 50%{transform:translate(90px,70px)} }
@keyframes float2 { 0%,100%{transform:translate(0,0)} 50%{transform:translate(-80px,-60px)} }

.hero { text-align:center; padding: 10px 0 18px; animation: rise .9s ease both; }
.hero h1 { font-size: clamp(2.4rem,5vw,4.2rem); font-weight:800; letter-spacing:-1px; margin:0;
  background: linear-gradient(90deg,#fff,#9d8bff,#00d4ff,#ff7ad9,#fff); background-size:300% auto;
  -webkit-background-clip:text; background-clip:text; color:transparent; animation: shine 7s linear infinite; }
.hero p { color:#a9a9d6; font-size:1.1rem; margin:.3rem 0 0; letter-spacing:.5px; }
.pill { display:inline-block; margin-top:12px; padding:5px 14px; border-radius:30px; font-size:.78rem; letter-spacing:2px;
  color:#00d4ff; border:1px solid rgba(0,212,255,.4); background:rgba(0,212,255,.08); animation: pulse 2.6s infinite; }
@keyframes shine { to { background-position: 300% center; } }
@keyframes rise { from{opacity:0; transform:translateY(24px)} to{opacity:1; transform:none} }
@keyframes pulse { 0%,100%{box-shadow:0 0 0 0 rgba(0,212,255,.45)} 50%{box-shadow:0 0 0 10px rgba(0,212,255,0)} }

.glass { background: rgba(255,255,255,.045); border:1px solid rgba(255,255,255,.1); border-radius:22px;
  padding:18px 20px; backdrop-filter: blur(14px); box-shadow: 0 10px 40px rgba(0,0,0,.35); animation: rise .8s ease both; }
.sec { font-family:'Space Grotesk'; font-weight:700; font-size:1.05rem; letter-spacing:2px; color:#b9acff;
  text-transform:uppercase; margin: 4px 0 8px; }

/* image stage */
.stage { position:relative; border-radius:26px; overflow:hidden; aspect-ratio:16/9; background:#000;
  border:1px solid rgba(255,255,255,.14); animation: rise .8s ease both;
  box-shadow: 0 25px 80px -15px var(--acc), 0 0 0 1px rgba(255,255,255,.04) inset; }
.stage img { width:100%; height:100%; object-fit:cover; display:block; animation: kenburns 18s ease-in-out infinite alternate; }
@keyframes kenburns { from{transform:scale(1.0)} to{transform:scale(1.12) translate(-1.5%,-1%)} }
.stage::after { content:""; position:absolute; inset:0; background:
  linear-gradient(115deg, transparent 38%, rgba(255,255,255,.22) 50%, transparent 62%);
  transform: translateX(-120%); animation: sweep 5.5s ease-in-out infinite; pointer-events:none; }
@keyframes sweep { 0%{transform:translateX(-120%)} 55%,100%{transform:translateX(120%)} }
.stage .grad { position:absolute; inset:auto 0 0 0; height:45%; background:linear-gradient(transparent,rgba(5,5,15,.92)); z-index:2; }
.stage .name { position:absolute; left:26px; bottom:20px; z-index:3; }
.stage .name small { color:var(--acc); letter-spacing:4px; font-weight:600; text-transform:uppercase; font-size:.8rem; }
.stage .name div { font-size:clamp(1.6rem,3.2vw,2.8rem); font-weight:800; line-height:1.05; color:#fff; }
.badge4k { position:absolute; top:16px; right:16px; z-index:3; padding:5px 12px; border-radius:10px; font-weight:800;
  font-size:.8rem; letter-spacing:1.5px; background:rgba(0,0,0,.55); border:1px solid rgba(255,255,255,.4); color:#fff; backdrop-filter:blur(6px); }
.chips { display:flex; flex-wrap:wrap; gap:9px; margin-top:14px; }
.chip { padding:8px 14px; border-radius:14px; background:rgba(255,255,255,.06); border:1px solid rgba(255,255,255,.1);
  font-size:.88rem; color:#d9d9ff; transition: .25s; animation: rise .9s ease both; }
.chip:hover { transform:translateY(-4px); border-color:var(--acc); background:rgba(255,255,255,.1); }
.chip b { color:#fff; }

/* widgets */
label, .stSelectbox label, .stSlider label, .stNumberInput label, .stRadio label p { color:#c8c8ee !important; font-weight:600 !important; }
div[data-baseweb="select"] > div, .stNumberInput input, .stTextInput input {
  background: rgba(255,255,255,.06) !important; border:1px solid rgba(255,255,255,.14) !important; border-radius:12px !important; color:#fff !important; }
div[data-baseweb="select"] > div:hover, .stTextInput input:focus { border-color:#7c5cff !important; box-shadow:0 0 0 3px rgba(124,92,255,.25) !important; }
.stButton > button, .stFormSubmitButton > button { position:relative; overflow:hidden; width:100%; border:none; border-radius:16px; padding:.85rem 1rem;
  font-size:1.1rem; font-weight:800; letter-spacing:1px; color:#fff;
  background: linear-gradient(90deg,#7c5cff,#00c6ff,#ff4fd8,#7c5cff); background-size:300% 100%; animation: shine 6s linear infinite;
  box-shadow: 0 10px 30px -8px rgba(124,92,255,.8); transition: transform .2s, box-shadow .2s; }
.stButton > button:hover, .stFormSubmitButton > button:hover { transform: translateY(-3px) scale(1.015); box-shadow:0 18px 40px -8px rgba(0,198,255,.7); color:#fff; border:none; }
.stButton > button:active { transform: scale(.98); }
.stTabs [data-baseweb="tab"] { font-weight:700; color:#a9a9d6; }
.stTabs [aria-selected="true"] { color:#fff !important; }

/* order receipt & garage */
.receipt { border-radius:24px; padding:22px 26px; border:1px solid rgba(61,220,132,.5); animation: pop .7s cubic-bezier(.2,1.4,.4,1) both;
  background: linear-gradient(135deg, rgba(61,220,132,.14), rgba(0,212,255,.08)); }
@keyframes pop { from{opacity:0; transform:scale(.88)} to{opacity:1; transform:none} }
.receipt h2 { margin:0 0 6px; color:#3ddc84; }
.receipt .row { display:flex; justify-content:space-between; padding:6px 0; border-bottom:1px dashed rgba(255,255,255,.14); color:#d9d9ff; }
.gcard { border-radius:20px; overflow:hidden; background:rgba(255,255,255,.05); border:1px solid rgba(255,255,255,.1); transition:.3s; }
.gcard:hover { transform: translateY(-8px) rotate(-.4deg); box-shadow:0 20px 50px -10px rgba(124,92,255,.6); }
.gcard img { width:100%; aspect-ratio:16/9; object-fit:cover; display:block; }
.gcard .gb { padding:12px 16px 14px; }
.gcard .gb b { font-size:1.1rem; } .gcard .gb span { color:#a9a9d6; font-size:.85rem; display:block; }
.stage-wrap { position:relative; }
</style>
<div class="orb o1"></div><div class="orb o2"></div><div class="orb o3"></div>
"""), unsafe_allow_html=True)

# ───────────────────────── state ─────────────────────────
st.session_state.setdefault("garage", [])
st.session_state.setdefault("last_order", None)
st.session_state.setdefault("celebrate", False)

# ───────────────────────── hero ─────────────────────────
st.markdown(h("""
<div class="hero">
<h1>AutoVerse</h1>
<p>Predict the fair price of any used car — then drive it home.</p>
<span class="pill">AI PRICE ENGINE · R² 0.92 · 4K SHOWROOM</span>
</div>
"""), unsafe_allow_html=True)

if st.session_state.celebrate:
    st.balloons()
    st.session_state.celebrate = False

# ───────────────────────── inputs + showcase ─────────────────────────
left, right = st.columns([5, 7], gap="large")

with left:
    st.markdown('<div class="sec">① Pick your car</div>', unsafe_allow_html=True)
    with st.container(border=False):
        c1, c2 = st.columns(2)
        brand = c1.selectbox("Brand", BRANDS, index=BRANDS.index("Kia"))
        models = [m for m, v in CARS.items() if v[0] == brand]
        model_name = c2.selectbox("Model", models, key=f"model_{brand}")

    spec = CARS[model_name]
    _, wiki_titles, d_engine, d_bhp, d_mileage, d_seats, d_fuel = spec
    k = model_name  # widgets re-key per model so specs auto-reset to that car's defaults

    t1, t2 = st.tabs(["⚙️  Condition", "🔧  Specs"])
    with t1:
        c1, c2 = st.columns(2)
        year = c1.slider("Manufacturing year", 2010, CURRENT_YEAR - 1, 2022, key=f"y_{k}")
        owner = c2.selectbox("Owner", ["First", "Second", "Third", "Fourth & Above"], key=f"o_{k}")
        km = st.slider("Kilometres driven", 1000, 250000, 35000, step=1000, key=f"km_{k}", format="%d km")
        c1, c2 = st.columns(2)
        fuel_opts = ["Petrol", "Diesel", "CNG", "Electric"]
        fuel = c1.selectbox("Fuel", fuel_opts, index=fuel_opts.index(d_fuel), key=f"f_{k}")
        trans = c2.selectbox("Transmission", ["Manual", "Automatic"], key=f"t_{k}")
    with t2:
        c1, c2 = st.columns(2)
        engine = c1.number_input("Engine (cc)", 600, 5000, int(d_engine), step=50, key=f"e_{k}")
        bhp = c2.number_input("Power (bhp)", 30.0, 600.0, float(d_bhp), step=1.0, key=f"p_{k}")
        c1, c2 = st.columns(2)
        mileage = c1.number_input("Mileage (km/l)", 5.0, 40.0, float(d_mileage), step=0.1, key=f"m_{k}")
        seat_opts = [2, 4, 5, 6, 7, 8]
        seats = c2.selectbox("Seats", seat_opts, index=seat_opts.index(d_seats), key=f"s_{k}")

inp = dict(brand=brand, model=model_name, year=year, km=km, fuel=fuel, transmission=trans,
           owner=owner, engine=engine, bhp=bhp, mileage=mileage, seats=seats)
try:
    price = predict_price(inp)
except Exception as exc:  # keep the UI alive if a pickle is missing
    st.error(f"Could not run the model: {exc}")
    st.stop()

accent = ACCENT.get(brand, "#7c5cff")
img_url = fetch_car_image(tuple(wiki_titles))
fallback = car_svg(accent)
img_src = img_url or fallback

with right:
    st.markdown('<div class="sec">② Your showroom</div>', unsafe_allow_html=True)
    st.markdown(h(f"""
    <div class="stage" style="--acc:{accent}">
    <img src="{img_src}" alt="{brand} {model_name}" referrerpolicy="no-referrer" onerror="this.onerror=null;this.src='{fallback}'">
    <span class="badge4k">4K UHD</span>
    <div class="grad"></div>
    <div class="name"><small>{brand}</small><div>{model_name}</div></div>
    </div>
    <div class="chips" style="--acc:{accent}">
    <span class="chip">📅 <b>{year}</b></span>
    <span class="chip">🛣️ <b>{km:,} km</b></span>
    <span class="chip">⛽ <b>{fuel}</b></span>
    <span class="chip">🕹️ <b>{trans}</b></span>
    <span class="chip">👤 <b>{owner} owner</b></span>
    <span class="chip">🔩 <b>{engine} cc</b></span>
    <span class="chip">⚡ <b>{bhp:.0f} bhp</b></span>
    <span class="chip">🌿 <b>{mileage} km/l</b></span>
    <span class="chip">💺 <b>{seats} seats</b></span>
    </div>
    """), unsafe_allow_html=True)

# ───────────────────────── animated price ─────────────────────────
st.markdown("<div style='height:14px'></div>", unsafe_allow_html=True)
low, high = price * 0.93, price * 1.07
PRICE_HTML = h(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@400;800&display=swap');
body{{margin:0;font-family:'Outfit',sans-serif;background:transparent;color:#fff}}
.wrap{{display:flex;align-items:center;justify-content:space-between;gap:20px;flex-wrap:wrap;padding:22px 30px;border-radius:26px;
border:1px solid rgba(255,255,255,.14);background:linear-gradient(120deg,rgba(124,92,255,.22),rgba(0,212,255,.10));
box-shadow:0 0 60px -10px {accent}88;position:relative;overflow:hidden}}
.wrap:before{{content:"";position:absolute;width:260px;height:260px;border-radius:50%;background:{accent};filter:blur(90px);opacity:.25;right:-60px;top:-90px;animation:p 4s ease-in-out infinite}}
@keyframes p{{50%{{transform:scale(1.4);opacity:.4}}}}
.l small{{letter-spacing:4px;color:#b9acff;font-weight:600}}
.price{{font-size:clamp(2.6rem,7vw,4.6rem);font-weight:800;line-height:1.05;background:linear-gradient(90deg,#fff,{accent});-webkit-background-clip:text;background-clip:text;color:transparent}}
.price .u{{font-size:1.4rem;font-weight:400;color:#c8c8ee;-webkit-text-fill-color:#c8c8ee}}
.r{{text-align:right;color:#c8c8ee;font-size:1.05rem;position:relative}}
.r b{{color:#fff;font-size:1.3rem}}
.bar{{height:8px;border-radius:8px;background:rgba(255,255,255,.12);margin-top:10px;overflow:hidden;width:260px;max-width:100%}}
.bar i{{display:block;height:100%;width:0;background:linear-gradient(90deg,#7c5cff,#00d4ff,#3ddc84);border-radius:8px;animation:g 1.6s .3s cubic-bezier(.2,.8,.2,1) forwards}}
@keyframes g{{to{{width:{min(100, price/45*100):.0f}%}}}}
</style>
<div class="wrap"><div class="l"><small>ESTIMATED MARKET VALUE</small>
<div class="price">₹ <span id="n">0.00</span> <span class="u">Lakh</span></div>
<div class="bar"><i></i></div></div>
<div class="r">Fair range<br><b>₹{low:.2f} – ₹{high:.2f} L</b><br>≈ ₹ {price*1e5:,.0f}</div></div>
<script>
const T={price:.4f},el=document.getElementById('n'),t0=performance.now(),D=1400;
(function f(t){{let p=Math.min((t-t0)/D,1);p=1-Math.pow(1-p,4);el.textContent=(T*p).toFixed(2);if(p<1)requestAnimationFrame(f)}})(t0);
</script>
""")
if hasattr(st, "iframe"):          # Streamlit ≥ 1.5x
    st.iframe(PRICE_HTML, height=175)
else:                              # older Streamlit
    components.html(PRICE_HTML, height=175)


# ───────────────────────── buy flow ─────────────────────────
def emi(principal, rate, months):
    r = rate / 12 / 100
    return principal * r * (1 + r) ** months / ((1 + r) ** months - 1) if principal > 0 else 0.0


@st.dialog("🛒  Complete your purchase", width="large")
def checkout():
    st.markdown(h(f"""
    <div class="gcard"><img src="{img_src}" onerror="this.onerror=null;this.src='{fallback}'">
    <div class="gb"><b>{year} {brand} {model_name}</b>
    <span>{fuel} · {trans} · {km:,} km · {owner} owner</span>
    <span style="color:#fff;font-size:1.3rem;font-weight:800;margin-top:4px">₹ {price:.2f} Lakh</span></div></div>
    """), unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    name = c1.text_input("Full name")
    phone = c2.text_input("Phone number")
    c1, c2 = st.columns(2)
    city = c1.text_input("City")
    pay = c2.radio("Payment", ["Full payment", "Car loan (EMI)"], horizontal=True)
    down, tenure, rate = price, 0, 0.0
    if pay.startswith("Car loan"):
        c1, c2, c3 = st.columns(3)
        down_pct = c1.slider("Down payment %", 10, 80, 20)
        tenure = c2.select_slider("Tenure (months)", [12, 24, 36, 48, 60, 72, 84], value=60)
        rate = c3.slider("Interest % p.a.", 7.0, 16.0, 9.5, step=0.25)
        down = price * down_pct / 100
        monthly = emi((price - down) * 1e5, rate, tenure)
        st.info(f"Down payment **₹ {down:.2f} L** · EMI **₹ {monthly:,.0f} / month** × {tenure} months")
    if st.button("✅  Confirm & buy now", key="confirm"):
        if not name.strip() or not phone.strip():
            st.warning("Please enter your name and phone number.")
            return
        order = dict(
            id="AV-" + "".join(random.choices(string.ascii_uppercase + string.digits, k=7)),
            name=name.strip(), phone=phone.strip(), city=city.strip() or "—",
            car=f"{year} {brand} {model_name}", price=price, pay=pay, down=down, tenure=tenure,
            emi=emi((price - down) * 1e5, rate, tenure) if tenure else 0, img=img_src,
            fallback=fallback, spec=f"{fuel} · {trans} · {km:,} km",
            date=dt.datetime.now().strftime("%d %b %Y, %I:%M %p"),
            delivery=(dt.date.today() + dt.timedelta(days=7)).strftime("%d %b %Y"))
        st.session_state.garage.insert(0, order)
        st.session_state.last_order = order
        st.session_state.celebrate = True
        st.rerun()


st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)
b1, b2, b3 = st.columns([3, 4, 3])
with b2:
    if st.button("🛒  BUY THIS CAR", key="buy"):
        checkout()

# ───────────────────────── receipt ─────────────────────────
o = st.session_state.last_order
if o:
    st.markdown("<div style='height:18px'></div>", unsafe_allow_html=True)
    ci, cr = st.columns([6, 5], gap="large")
    with ci:
        st.markdown(h(f"""
        <div class="stage" style="--acc:#3ddc84"><img src="{o['img']}" onerror="this.onerror=null;this.src='{o['fallback']}'">
        <span class="badge4k">SOLD TO YOU 🎉</span><div class="grad"></div>
        <div class="name"><small>Congratulations {o['name'].split()[0]}</small><div>{o['car']}</div></div></div>
        """), unsafe_allow_html=True)
    with cr:
        pay_rows = (f"<div class='row'><span>Down payment</span><b>₹ {o['down']:.2f} L</b></div>"
                    f"<div class='row'><span>EMI</span><b>₹ {o['emi']:,.0f} × {o['tenure']} mo</b></div>"
                    if o["tenure"] else "")
        st.markdown(h(f"""
        <div class="receipt"><h2>✔ Order confirmed</h2>
        <div class="row"><span>Order ID</span><b>{o['id']}</b></div>
        <div class="row"><span>Car</span><b>{o['car']}</b></div>
        <div class="row"><span>Specs</span><b>{o['spec']}</b></div>
        <div class="row"><span>Price</span><b>₹ {o['price']:.2f} Lakh</b></div>
        <div class="row"><span>Payment</span><b>{o['pay']}</b></div>
        {pay_rows}
        <div class="row"><span>Buyer</span><b>{o['name']} · {o['city']}</b></div>
        <div class="row"><span>Ordered</span><b>{o['date']}</b></div>
        <div class="row" style="border:none"><span>Estimated delivery</span><b>🚚 {o['delivery']}</b></div></div>
        """), unsafe_allow_html=True)

# ───────────────────────── garage ─────────────────────────
if st.session_state.garage:
    st.markdown("<div style='height:26px'></div><div class='sec'>🏁 My garage</div>", unsafe_allow_html=True)
    cols = st.columns(3)
    for i, g in enumerate(st.session_state.garage):
        with cols[i % 3]:
            st.markdown(h(f"""
            <div class="gcard"><img src="{g['img']}" onerror="this.onerror=null;this.src='{g['fallback']}'">
            <div class="gb"><b>{g['car']}</b><span>{g['spec']}</span>
            <span>₹ {g['price']:.2f} L · {g['id']}</span></div></div>
            """), unsafe_allow_html=True)

st.markdown("<p style='text-align:center;color:#6d6d99;margin-top:40px;font-size:.8rem'>"
            "Prices are AI estimates from a regression model · Photos via Wikimedia Commons</p>",
            unsafe_allow_html=True)
