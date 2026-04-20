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

CROPS = ["rice","wheat","maize","sugarcane","cotton","tobacco","barley","groundnut","mustard","coffee","jute","rubber","tea","coconut","potato","onion","tomato","soybean","sunflower","banana"]
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

@app.get("/", response_class=HTMLResponse)
def home():
    return f"""
<!DOCTYPE html>
<html>
<head>
<title>Smart Fertilizer Advisor</title>
<style>
body {{
  margin:0;
  font-family:Arial;
  color:white;
  transition: background 1s ease-in-out;
}}
.container {{
  max-width:900px;
  margin:auto;
  padding:20px;
  background:rgba(0,0,0,0.6);
  border-radius:12px;
  margin-top:40px;
}}
h1 {{
  text-align:center;
}}
input,select {{
  width:100%;
  padding:10px;
  margin:6px 0;
  border-radius:6px;
  border:none;
}}
button {{
  width:100%;
  padding:12px;
  background:#2ecc71;
  color:white;
  border:none;
  border-radius:8px;
  font-size:16px;
}}
.result {{
  margin-top:20px;
  padding:15px;
  background:#27ae60;
  border-radius:8px;
  text-align:center;
  font-size:20px;
}}
</style>
</head>

<body>

<div class="container">
<h1>Smart Fertilizer Advisor</h1>

<select id="crop">
{''.join([f'<option>{c}</option>' for c in CROPS])}
</select>

<select id="soil">
{''.join([f'<option>{s}</option>' for s in SOILS])}
</select>

<input id="temp" type="range" min="10" max="50" value="25" oninput="tval.innerText=this.value">
Temperature: <span id="tval">25</span>

<input id="moist" type="range" min="0" max="1" step="0.01" value="0.6" oninput="mval.innerText=this.value">
Moisture: <span id="mval">0.6</span>

<input id="rain" type="range" min="0" max="500" value="200" oninput="rval.innerText=this.value">
Rainfall: <span id="rval">200</span>

<input id="ph" type="range" min="3" max="10" step="0.1" value="6.5" oninput="phval.innerText=this.value">
pH: <span id="phval">6.5</span>

<input id="n" type="range" min="0" max="200" value="75" oninput="nval.innerText=this.value">
Nitrogen: <span id="nval">75</span>

<input id="p" type="range" min="0" max="200" value="80" oninput="pval.innerText=this.value">
Phosphorous: <span id="pval">80</span>

<input id="k" type="range" min="0" max="200" value="90" oninput="kval.innerText=this.value">
Potassium: <span id="kval">90</span>

<input id="c" type="range" min="0" max="5" step="0.1" value="1.5" oninput="cval.innerText=this.value">
Carbon: <span id="cval">1.5</span>

<button onclick="predict()">Predict Fertilizer</button>

<div id="result" class="result"></div>
</div>

<script>
const images = [
"https://images.unsplash.com/photo-1500382017468-9049fed747ef",
"https://images.unsplash.com/photo-1464226184884-fa280b87c399",
"https://images.unsplash.com/photo-1501004318641-b39e6451bec6",
"https://images.unsplash.com/photo-1492496913980-501348b61469",
"https://images.unsplash.com/photo-1472396961693-142e6e269027"
];

let index = 0;
function changeBackground() {{
  document.body.style.background = "url(" + images[index] + ") center/cover no-repeat";
  index = (index + 1) % images.length;
}}
setInterval(changeBackground, 5000);
changeBackground();

async function predict() {{
  const res = await fetch("/predict", {{
    method:"POST",
    headers:{{"Content-Type":"application/json"}},
    body:JSON.stringify({{
      Temperature:parseFloat(temp.value),
      Moisture:parseFloat(moist.value),
      Rainfall:parseFloat(rain.value),
      PH:parseFloat(ph.value),
      Nitrogen:parseFloat(n.value),
      Phosphorous:parseFloat(p.value),
      Potassium:parseFloat(k.value),
      Carbon:parseFloat(c.value),
      Soil:soil.value,
      Crop:crop.value
    }})
  }});
  const data = await res.json();
  document.getElementById("result").innerText = data.recommended_fertilizer || data.error;
}}
</script>

</body>
</html>
"""

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