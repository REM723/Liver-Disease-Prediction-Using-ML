import csv
from pathlib import Path
from statistics import median

from flask import Flask, request, render_template_string
from sklearn.ensemble import RandomForestClassifier

HERE = Path(__file__).parent
FIELDS = [
    ("Age", "Age", "years", 45),
    ("Total_Bilirubin", "Total bilirubin", "mg/dL", 1.0),
    ("Direct_Bilirubin", "Direct bilirubin", "mg/dL", 0.3),
    ("Alkaline_Phosphotase", "Alkaline phosphatase", "IU/L", 200),
    ("Alamine_Aminotransferase", "Alanine aminotransferase (ALT)", "IU/L", 35),
    ("Aspartate_Aminotransferase", "Aspartate aminotransferase (AST)", "IU/L", 40),
    ("Total_Protiens", "Total proteins", "g/dL", 6.8),
    ("Albumin", "Albumin", "g/dL", 3.3),
    ("Albumin_and_Globulin_Ratio", "Albumin / globulin ratio", "", 1.0),
]
NUM_KEYS = [f[0] for f in FIELDS if f[0] != "Gender"]


def load_train():
    rows = list(csv.DictReader(open(HERE / "liver.csv")))
    # median fill for the one column with blanks (A/G ratio)
    med = {k: median(float(r[k]) for r in rows if r[k]) for k in NUM_KEYS}
    X, y = [], []
    for r in rows:
        feat = [1 if r["Gender"] == "Male" else 0]
        feat += [float(r[k]) if r[k] else med[k] for k in NUM_KEYS]
        X.append(feat)
        y.append(1 if r["Dataset"] == "1" else 0)  # 1 = disease in ILPD
    clf = RandomForestClassifier(n_estimators=300, random_state=0, class_weight="balanced")
    clf.fit(X, y)
    return clf, med


MODEL, MED = load_train()

PAGE = """<!doctype html><html lang=en><head><meta charset=utf-8>
<meta name=viewport content="width=device-width,initial-scale=1">
<title>Hepatic — liver risk estimate</title>
<link rel=preconnect href=https://fonts.googleapis.com>
<link href="https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,400;9..144,600&family=Inter:wght@400;500&display=swap" rel=stylesheet>
<style>
:root{--ink:#1c1a17;--paper:#f4f1ea;--line:#d9d3c7;--accent:#7c3b2e;--muted:#7a7266}
*{box-sizing:border-box;margin:0}
body{background:var(--paper);color:var(--ink);font:16px/1.55 Inter,sans-serif;
  -webkit-font-smoothing:antialiased}
.wrap{max-width:760px;margin:0 auto;padding:64px 24px 96px}
.eyebrow{font-size:12px;letter-spacing:.22em;text-transform:uppercase;color:var(--muted)}
h1{font:600 clamp(38px,7vw,60px)/1.02 Fraunces,serif;letter-spacing:-.02em;margin:14px 0 10px}
.lede{max-width:46ch;color:var(--muted)}
hr{border:0;border-top:1px solid var(--line);margin:40px 0}
form{display:grid;grid-template-columns:1fr 1fr;gap:22px 28px}
.field{display:flex;flex-direction:column;gap:6px}
.field.wide{grid-column:1/-1}
label{font-size:13px;color:var(--muted)}
label b{color:var(--ink);font-weight:500}
input,select{font:15px Inter,sans-serif;color:var(--ink);background:transparent;
  border:0;border-bottom:1px solid var(--line);padding:8px 2px;outline:none}
input:focus,select:focus{border-color:var(--accent)}
.seg{display:flex;gap:0;border:1px solid var(--line);border-radius:999px;overflow:hidden;width:max-content}
.seg label{padding:9px 22px;cursor:pointer;color:var(--muted);font-size:14px}
.seg input{display:none}
.seg input:checked+span{background:var(--ink);color:var(--paper)}
.seg span{padding:9px 22px;display:block;margin:-9px -22px;border-radius:999px}
button{grid-column:1/-1;justify-self:start;margin-top:8px;background:var(--ink);color:var(--paper);
  border:0;border-radius:999px;padding:15px 40px;font:500 15px Inter;cursor:pointer;letter-spacing:.01em}
button:hover{background:var(--accent)}
.result{grid-column:1/-1;border-top:1px solid var(--line);padding-top:28px;margin-top:8px;
  display:flex;align-items:baseline;gap:20px;flex-wrap:wrap}
.pct{font:600 56px/1 Fraunces,serif;letter-spacing:-.02em}
.verdict{font-size:14px;color:var(--muted);max-width:38ch}
.tag{font-size:12px;letter-spacing:.14em;text-transform:uppercase;padding:5px 12px;border-radius:999px}
.hi{background:#efe1dd;color:var(--accent)}
.lo{background:#e2e8dd;color:#3d5a3d}
.note{margin-top:44px;font-size:12.5px;color:var(--muted);max-width:52ch}
</style></head><body><div class=wrap>
<div class=eyebrow>Hepatic · clinical estimate</div>
<h1>Liver&nbsp;risk, read&nbsp;from bloodwork.</h1>
<p class=lede>Enter a standard liver panel. A trained model returns the estimated likelihood of hepatic disease — not a diagnosis.</p>
<hr>
<form method=post>
<div class="field wide">
  <label>Sex</label>
  <div class=seg>
    <label><input type=radio name=Gender value=Male {{'checked' if v.Gender!='Female'}}><span>Male</span></label>
    <label><input type=radio name=Gender value=Female {{'checked' if v.Gender=='Female'}}><span>Female</span></label>
  </div>
</div>
{% for key,label,unit,ph in fields %}
<div class=field>
  <label><b>{{label}}</b>{% if unit %} · {{unit}}{% endif %}</label>
  <input name="{{key}}" type=number step=any required value="{{ v[key] }}" placeholder="{{ph}}">
</div>
{% endfor %}
<button>Estimate risk</button>
{% if pct is not none %}
<div class=result>
  <div class=pct>{{pct}}%</div>
  <div>
    <span class="tag {{'hi' if pct>=50 else 'lo'}}">{{'Elevated signal' if pct>=50 else 'Low signal'}}</span>
    <p class=verdict>Estimated probability of liver disease given this panel.</p>
  </div>
</div>
{% endif %}
</form>
<p class=note>Educational tool trained on the Indian Liver Patient Dataset. It is not medical advice and cannot diagnose. Consult a clinician for interpretation.</p>
</div></body></html>"""


app = Flask(__name__)


@app.route("/", methods=["GET", "POST"])
def home():
    v = {k: request.form.get(k, "") for k in [f[0] for f in FIELDS] + ["Gender"]}
    pct = None
    if request.method == "POST":
        feat = [1 if v["Gender"] == "Male" else 0]
        feat += [float(v[k]) if v[k] else MED[k] for k in NUM_KEYS]
        pct = round(MODEL.predict_proba([feat])[0][1] * 100)
    return render_template_string(PAGE, fields=FIELDS, v=v, pct=pct)


if __name__ == "__main__":
    app.run(debug=True, port=5000)
