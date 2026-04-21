from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import pandas as pd
from joblib import load

model = load("fertilizer_recommendation.joblib")
scaler = load("fertilizer_scaler.joblib")
fertilizer_encoder = load("fertilizer_encoder.joblib")
crop_encoder = load("crop_encoder.joblib")
feature_names = load("fertilizer_feature_names.joblib")

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

CROPS = ["rice","wheat","maize","sugarcane","cotton","tobacco","barley","groundnut",
         "mustard","coffee","jute","rubber","tea","coconut","potato","onion",
         "tomato","soybean","sunflower","banana"]
SOILS = ["Loamy Soil","Acidic Soil","Alkaline Soil","Peaty Soil","Neutral Soil"]

class InputData(BaseModel):
    Temperature: float
    Moisture: float
    Rainfall: float
    PH: float
    Nitrogen: float
    Phosphorous: float
    Potassium: float
    Carbon: float
    Soil: str
    Crop: str

def encode_soil(soil):
    return {
        'Acidic_Soil': 1 if soil == 'Acidic Soil' else 0,
        'Alkaline_Soil': 1 if soil == 'Alkaline Soil' else 0,
        'Loamy_Soil': 1 if soil == 'Loamy Soil' else 0,
        'Neutral_Soil': 1 if soil == 'Neutral Soil' else 0,
        'Peaty_Soil': 1 if soil == 'Peaty Soil' else 0,
    }

CROP_OPTIONS = "".join([f'<option>{c}</option>' for c in CROPS])
SOIL_OPTIONS = "".join([f'<option>{s}</option>' for s in SOILS])

HTML_PAGE = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Smart Fertilizer Advisor</title>
<style>
*{{box-sizing:border-box;margin:0;padding:0}}
:root{{
  --g1:#1a4a1a;--g2:#2d7a2d;--g3:#3d9e3d;--g4:#52c452;--g5:#7dd87d;
  --g-bg:#f0f9f0;--g-card:#ffffff;
  --text:#1a2e1a;--text2:#4a6a4a;--text3:#7a9a7a;
  --border:#c8e6c8;--radius:12px;
}}
body{{background:var(--g-bg);font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Arial,sans-serif;color:var(--text);min-height:100vh}}
.hero{{background:linear-gradient(135deg,#1a4a1a 0%,#2d7a2d 50%,#1a5c1a 100%);padding:2.5rem 1.5rem 2rem;text-align:center;position:relative;overflow:hidden}}
.hero::before{{content:'';position:absolute;inset:0;background-image:radial-gradient(circle at 20% 80%,rgba(255,255,255,0.04) 0%,transparent 50%),radial-gradient(circle at 80% 20%,rgba(255,255,255,0.04) 0%,transparent 50%)}}
.hero-badge{{display:inline-flex;align-items:center;gap:6px;background:rgba(255,255,255,0.12);border:1px solid rgba(255,255,255,0.2);color:rgba(255,255,255,0.9);font-size:12px;padding:4px 14px;border-radius:20px;margin-bottom:1rem;letter-spacing:0.02em}}
.hero h1{{color:#fff;font-size:26px;font-weight:500;margin-bottom:8px;letter-spacing:-0.01em}}
.hero p{{color:rgba(255,255,255,0.6);font-size:14px}}
.main{{padding:1.5rem;max-width:600px;margin:0 auto}}
.status-bar{{display:flex;align-items:center;gap:8px;margin-bottom:1.25rem}}
.status-dot{{width:8px;height:8px;border-radius:50%;background:var(--g4);flex-shrink:0;animation:pulse 2s ease-in-out infinite}}
@keyframes pulse{{0%,100%{{opacity:1}}50%{{opacity:0.4}}}}
.status-text{{font-size:12px;color:var(--text3)}}
.tab-row{{display:flex;gap:4px;margin-bottom:1.25rem;background:#e8f5e8;border-radius:10px;padding:3px}}
.tab{{flex:1;padding:8px;text-align:center;font-size:13px;border-radius:7px;cursor:pointer;color:var(--text2);border:none;background:none;font-family:inherit;transition:all 0.15s}}
.tab.active{{background:#fff;color:var(--g1);font-weight:500;border:0.5px solid var(--border);box-shadow:0 1px 3px rgba(0,0,0,0.06)}}
.section{{display:none}}.section.active{{display:block}}
.card{{background:var(--g-card);border:0.5px solid var(--border);border-radius:var(--radius);padding:1.25rem;margin-bottom:1rem}}
.card-header{{display:flex;align-items:center;gap:10px;margin-bottom:1rem;padding-bottom:0.875rem;border-bottom:0.5px solid var(--border)}}
.card-icon{{width:34px;height:34px;border-radius:9px;display:flex;align-items:center;justify-content:center;background:#eaf6ea;font-size:16px;flex-shrink:0}}
.card-title{{font-size:14px;font-weight:500;color:var(--text)}}
.card-sub{{font-size:12px;color:var(--text3);margin-top:2px}}
.field-label{{font-size:11px;font-weight:500;color:var(--text3);text-transform:uppercase;letter-spacing:0.07em;margin-bottom:0.5rem}}
.select-wrap{{position:relative;margin-bottom:1rem}}
.select-wrap:last-child{{margin-bottom:0}}
select{{width:100%;padding:10px 36px 10px 12px;border:0.5px solid var(--border);border-radius:8px;background:#f7fdf7;color:var(--text);font-size:14px;appearance:none;cursor:pointer;outline:none;font-family:inherit;transition:border-color 0.15s,box-shadow 0.15s}}
select:hover{{border-color:#a0cda0}}
select:focus{{border-color:var(--g3);box-shadow:0 0 0 3px rgba(61,158,61,0.12)}}
.select-arrow{{position:absolute;right:12px;top:50%;transform:translateY(-50%);pointer-events:none;color:var(--text3);font-size:10px}}
.npk-grid{{display:grid;grid-template-columns:1fr 1fr 1fr;gap:8px;margin-bottom:1.25rem}}
.npk-card{{background:#f7fdf7;border:0.5px solid var(--border);border-radius:9px;padding:12px;text-align:center}}
.npk-badge{{display:inline-block;width:22px;height:22px;border-radius:6px;font-size:11px;font-weight:500;line-height:22px;margin-bottom:6px}}
.npk-n{{background:#d4edda;color:#1a5c2a}}
.npk-p{{background:#fde8cc;color:#7a4010}}
.npk-k{{background:#cce0f5;color:#0a3a6a}}
.npk-val{{font-size:20px;font-weight:500;color:var(--g1)}}
.npk-unit{{font-size:10px;color:var(--text3);margin-top:1px}}
.slider-row{{margin-bottom:1.125rem}}
.slider-header{{display:flex;justify-content:space-between;align-items:center;margin-bottom:7px}}
.slider-name{{font-size:13px;color:var(--text2);font-weight:500}}
.slider-val{{font-size:13px;font-weight:500;color:var(--g2);min-width:44px;text-align:right;background:#eaf6ea;padding:2px 8px;border-radius:5px}}
input[type=range]{{width:100%;height:4px;border-radius:2px;appearance:none;-webkit-appearance:none;background:linear-gradient(to right,var(--g3) var(--pct,50%),#d4e8d4 var(--pct,50%));outline:none;cursor:pointer}}
input[type=range]::-webkit-slider-thumb{{-webkit-appearance:none;width:18px;height:18px;border-radius:50%;background:var(--g2);border:2.5px solid #fff;box-shadow:0 1px 4px rgba(0,0,0,0.2)}}
input[type=range]::-moz-range-thumb{{width:18px;height:18px;border-radius:50%;background:var(--g2);border:2.5px solid #fff;box-shadow:0 1px 4px rgba(0,0,0,0.2)}}
.env-grid{{display:grid;grid-template-columns:1fr 1fr;gap:8px;margin-bottom:1.25rem}}
.env-item{{background:#f7fdf7;border:0.5px solid var(--border);border-radius:9px;padding:10px 14px}}
.env-item-label{{font-size:11px;color:var(--text3);margin-bottom:3px}}
.env-item-val{{font-size:16px;font-weight:500;color:var(--g1)}}
.predict-btn{{width:100%;padding:14px;background:linear-gradient(135deg,#2d7a2d,#3d9e3d);color:#fff;border:none;border-radius:10px;font-size:15px;font-weight:500;cursor:pointer;display:flex;align-items:center;justify-content:center;gap:8px;transition:opacity 0.15s,transform 0.1s;margin-top:1.5rem;font-family:inherit;letter-spacing:0.01em}}
.predict-btn:hover{{opacity:0.92}}
.predict-btn:active{{transform:scale(0.98)}}
.predict-btn.loading{{opacity:0.65;cursor:default}}
.spinner{{width:16px;height:16px;border:2px solid rgba(255,255,255,0.3);border-top-color:#fff;border-radius:50%;animation:spin 0.7s linear infinite;display:none}}
.predict-btn.loading .spinner{{display:block}}
.predict-btn.loading .btn-icon{{display:none}}
@keyframes spin{{to{{transform:rotate(360deg)}}}}
.result-card{{background:linear-gradient(135deg,#1a4a1a,#2d7a2d);border-radius:var(--radius);padding:1.75rem;margin-top:1rem;text-align:center;display:none;animation:fadeSlide 0.3s ease}}
.result-card.show{{display:block}}
@keyframes fadeSlide{{from{{opacity:0;transform:translateY(8px)}}to{{opacity:1;transform:translateY(0)}}}}
.result-icon-wrap{{width:56px;height:56px;background:rgba(255,255,255,0.12);border-radius:16px;display:flex;align-items:center;justify-content:center;margin:0 auto 1rem;font-size:24px}}
.result-label{{font-size:11px;color:rgba(255,255,255,0.55);text-transform:uppercase;letter-spacing:0.1em;margin-bottom:8px}}
.result-name{{font-size:28px;font-weight:500;color:#fff;margin-bottom:8px;letter-spacing:-0.01em}}
.result-sub{{font-size:13px;color:rgba(255,255,255,0.55)}}
.result-divider{{width:40px;height:2px;background:rgba(255,255,255,0.2);border-radius:1px;margin:12px auto}}
.footer{{text-align:center;padding:1.5rem;color:var(--text3);font-size:12px}}
</style>
</head>
<body>

<div class="hero">
  <div class="hero-badge">&#127807; AI-Powered Advisory</div>
  <h1>Smart Fertilizer Advisor</h1>
  <p>Precision recommendations based on soil &amp; crop analysis</p>
</div>

<div class="main">
  <div class="status-bar">
    <div class="status-dot"></div>
    <span class="status-text">Model ready &mdash; adjust parameters and predict</span>
  </div>

  <div class="tab-row">
    <button class="tab active" onclick="switchTab('crop',this)">Crop &amp; Soil</button>
    <button class="tab" onclick="switchTab('nutrients',this)">Nutrients</button>
    <button class="tab" onclick="switchTab('env',this)">Environment</button>
  </div>

  <!-- Tab 1: Crop & Soil -->
  <div id="tab-crop" class="section active">
    <div class="card">
      <div class="card-header">
        <div class="card-icon">&#127807;</div>
        <div>
          <div class="card-title">Crop &amp; soil selection</div>
          <div class="card-sub">Choose your crop type and soil</div>
        </div>
      </div>
      <div class="field-label">Crop type</div>
      <div class="select-wrap">
        <select id="crop">{CROP_OPTIONS}</select>
        <span class="select-arrow">&#9660;</span>
      </div>
      <div class="field-label">Soil type</div>
      <div class="select-wrap" style="margin-bottom:0">
        <select id="soil">{SOIL_OPTIONS}</select>
        <span class="select-arrow">&#9660;</span>
      </div>
    </div>
  </div>

  <!-- Tab 2: Nutrients -->
  <div id="tab-nutrients" class="section">
    <div class="card">
      <div class="card-header">
        <div class="card-icon">&#9879;</div>
        <div>
          <div class="card-title">Soil nutrient levels</div>
          <div class="card-sub">NPK and organic carbon</div>
        </div>
      </div>
      <div class="npk-grid">
        <div class="npk-card">
          <div class="npk-badge npk-n">N</div>
          <div class="npk-val" id="nv">75</div>
          <div class="npk-unit">mg/kg</div>
        </div>
        <div class="npk-card">
          <div class="npk-badge npk-p">P</div>
          <div class="npk-val" id="pv">80</div>
          <div class="npk-unit">mg/kg</div>
        </div>
        <div class="npk-card">
          <div class="npk-badge npk-k">K</div>
          <div class="npk-val" id="kv">90</div>
          <div class="npk-unit">mg/kg</div>
        </div>
      </div>
      <div class="slider-row">
        <div class="slider-header">
          <span class="slider-name">Nitrogen (N)</span>
          <span class="slider-val" id="nd">75</span>
        </div>
        <input type="range" id="n" min="0" max="200" value="75" step="1"
          oninput="updNPK(this,'nd','nv')">
      </div>
      <div class="slider-row">
        <div class="slider-header">
          <span class="slider-name">Phosphorous (P)</span>
          <span class="slider-val" id="pd">80</span>
        </div>
        <input type="range" id="p" min="0" max="200" value="80" step="1"
          oninput="updNPK(this,'pd','pv')">
      </div>
      <div class="slider-row">
        <div class="slider-header">
          <span class="slider-name">Potassium (K)</span>
          <span class="slider-val" id="kd">90</span>
        </div>
        <input type="range" id="k" min="0" max="200" value="90" step="1"
          oninput="updNPK(this,'kd','kv')">
      </div>
      <div class="slider-row" style="margin-bottom:0">
        <div class="slider-header">
          <span class="slider-name">Organic carbon</span>
          <span class="slider-val" id="cd">1.5</span>
        </div>
        <input type="range" id="c" min="0" max="5" value="1.5" step="0.1"
          oninput="updDec(this,'cd',1)">
      </div>
    </div>
  </div>

  <!-- Tab 3: Environment -->
  <div id="tab-env" class="section">
    <div class="card">
      <div class="card-header">
        <div class="card-icon">&#127777;</div>
        <div>
          <div class="card-title">Environmental conditions</div>
          <div class="card-sub">Climate and soil pH readings</div>
        </div>
      </div>
      <div class="env-grid">
        <div class="env-item">
          <div class="env-item-label">Temperature</div>
          <div class="env-item-val" id="tv2">25&deg;C</div>
        </div>
        <div class="env-item">
          <div class="env-item-label">Moisture</div>
          <div class="env-item-val" id="mv2">0.60</div>
        </div>
        <div class="env-item">
          <div class="env-item-label">Rainfall</div>
          <div class="env-item-val" id="rv2">200 mm</div>
        </div>
        <div class="env-item">
          <div class="env-item-label">pH level</div>
          <div class="env-item-val" id="phv2">6.5</div>
        </div>
      </div>
      <div class="slider-row">
        <div class="slider-header">
          <span class="slider-name">Temperature</span>
          <span class="slider-val" id="td">25&deg;C</span>
        </div>
        <input type="range" id="temp" min="10" max="50" value="25" step="1"
          oninput="updEnv(this,'td','tv2','&deg;C')">
      </div>
      <div class="slider-row">
        <div class="slider-header">
          <span class="slider-name">Soil moisture</span>
          <span class="slider-val" id="md">0.60</span>
        </div>
        <input type="range" id="moist" min="0" max="1" step="0.01" value="0.6"
          oninput="updDec2(this,'md','mv2',2)">
      </div>
      <div class="slider-row">
        <div class="slider-header">
          <span class="slider-name">Rainfall</span>
          <span class="slider-val" id="rd">200 mm</span>
        </div>
        <input type="range" id="rain" min="0" max="500" value="200" step="1"
          oninput="updEnv(this,'rd','rv2',' mm')">
      </div>
      <div class="slider-row" style="margin-bottom:0">
        <div class="slider-header">
          <span class="slider-name">Soil pH</span>
          <span class="slider-val" id="phd">6.5</span>
        </div>
        <input type="range" id="ph" min="3" max="10" step="0.1" value="6.5"
          oninput="updDec2(this,'phd','phv2',1)">
      </div>
    </div>
  </div>

  <button class="predict-btn" id="btn" onclick="predict()">
    <span class="spinner" id="spinner"></span>
    
    <span id="btn-text">Predict fertilizer</span>
  </button>

  <div class="result-card" id="result">
    <div class="result-icon-wrap">&#127807;</div>
    <div class="result-label">Recommended fertilizer</div>
    <div class="result-name" id="result-name">&mdash;</div>
    <div class="result-divider"></div>
    <div class="result-sub" id="result-sub">Based on your soil and crop parameters</div>
  </div>
</div>

<div class="footer">Smart Fertilizer Advisor &mdash; Powered by ML</div>

<script>
function setPct(el){{
  const min=parseFloat(el.min),max=parseFloat(el.max),val=parseFloat(el.value);
  const pct=((val-min)/(max-min)*100).toFixed(1);
  el.style.setProperty('--pct',pct+'%');
}}
function updNPK(el,dispId,cardId){{
  const v=Math.round(parseFloat(el.value));
  document.getElementById(dispId).textContent=v;
  document.getElementById(cardId).textContent=v;
  setPct(el);
}}
function updDec(el,dispId,dec){{
  const v=parseFloat(el.value).toFixed(dec);
  document.getElementById(dispId).textContent=v;
  setPct(el);
}}
function updEnv(el,dispId,cardId,unit){{
  const v=Math.round(parseFloat(el.value));
  document.getElementById(dispId).innerHTML=v+unit;
  document.getElementById(cardId).innerHTML=v+unit;
  setPct(el);
}}
function updDec2(el,dispId,cardId,dec){{
  const v=parseFloat(el.value).toFixed(dec);
  document.getElementById(dispId).textContent=v;
  document.getElementById(cardId).textContent=v;
  setPct(el);
}}
document.querySelectorAll('input[type=range]').forEach(setPct);

function switchTab(t,btn){{
  document.querySelectorAll('.tab').forEach(b=>b.classList.remove('active'));
  document.querySelectorAll('.section').forEach(s=>s.classList.remove('active'));
  btn.classList.add('active');
  document.getElementById('tab-'+t).classList.add('active');
}}

async function predict(){{
  const btn=document.getElementById('btn');
  if(btn.classList.contains('loading')) return;
  btn.classList.add('loading');
  document.getElementById('btn-text').textContent='Analyzing...';
  const payload={{
    Temperature:parseFloat(document.getElementById('temp').value),
    Moisture:parseFloat(document.getElementById('moist').value),
    Rainfall:parseFloat(document.getElementById('rain').value),
    PH:parseFloat(document.getElementById('ph').value),
    Nitrogen:parseFloat(document.getElementById('n').value),
    Phosphorous:parseFloat(document.getElementById('p').value),
    Potassium:parseFloat(document.getElementById('k').value),
    Carbon:parseFloat(document.getElementById('c').value),
    Soil:document.getElementById('soil').value,
    Crop:document.getElementById('crop').value
  }};
  try{{
    const res=await fetch('/predict',{{
      method:'POST',
      headers:{{'Content-Type':'application/json'}},
      body:JSON.stringify(payload)
    }});
    const data=await res.json();
    const name=data.recommended_fertilizer||data.error||'Unknown';
    document.getElementById('result-name').textContent=name;
    document.getElementById('result-sub').textContent='Optimal for '+payload.Crop+' on '+payload.Soil;
    document.getElementById('result').classList.add('show');
    document.getElementById('result').scrollIntoView({{behavior:'smooth',block:'nearest'}});
  }}catch(e){{
    document.getElementById('result-name').textContent='Connection error';
    document.getElementById('result-sub').textContent='Could not reach the prediction server';
    document.getElementById('result').classList.add('show');
  }}finally{{
    btn.classList.remove('loading');
    document.getElementById('btn-text').textContent='Predict fertilizer';
  }}
}}
</script>
</body>
</html>"""

@app.get("/", response_class=HTMLResponse)
def home():
    return HTML_PAGE

@app.post("/predict")
def predict(data: InputData):
    soil_features = encode_soil(data.Soil)
    crop_encoded = crop_encoder.transform([data.Crop])[0]

    input_dict = {
        'Temperature': data.Temperature,
        'Moisture': data.Moisture,
        'Rainfall': data.Rainfall,
        'PH': data.PH,
        'Nitrogen': data.Nitrogen,
        'Phosphorous': data.Phosphorous,
        'Potassium': data.Potassium,
        'Carbon': data.Carbon,
        **soil_features,
        'Crops': crop_encoded
    }

    input_df = pd.DataFrame([input_dict])
    input_df = input_df[feature_names]
    input_scaled = scaler.transform(input_df)

    prediction = model.predict(input_scaled)
    fertilizer = fertilizer_encoder.inverse_transform(prediction)[0]

    return {"recommended_fertilizer": fertilizer}