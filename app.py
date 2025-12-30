import streamlit as st
import torch
from PIL import Image
from predict import load_model, preprocess_pil

# =============================
# Config
# =============================
st.set_page_config(
    page_title="Sektör Kampüste Proje",
    page_icon="✨",
    layout="wide",
    initial_sidebar_state="collapsed",
)

MODEL_PATH = "models/animals10_resnet18.pth"

LABEL_TR = {
    "cane": "Köpek",
    "cavallo": "At",
    "elefante": "Fil",
    "farfalla": "Kelebek",
    "gallina": "Tavuk",
    "gatto": "Kedi",
    "mucca": "İnek",
    "pecora": "Koyun",
    "ragno": "Örümcek",
    "scoiattolo": "Sincap",
}

METRICS = {
    "Accuracy": 0.9459,
    "Precision(macro)": 0.9436,
    "Recall(macro)": 0.9417,
    "F1(macro)": 0.9423,
}

OWNER = "Mert Bülbül — 220502006"


# =============================
# Device detect
# =============================
def detect_device():
    if torch.cuda.is_available():
        try:
            return "cuda", f"GPU ({torch.cuda.get_device_name(0)})"
        except Exception:
            return "cuda", "GPU (CUDA)"
    return "cpu", "CPU"


device, device_label = detect_device()

# =============================
# Style
# =============================
st.markdown(
    """
<style>
header, footer, #MainMenu {visibility:hidden;}
div[data-testid="stToolbar"]{visibility:hidden; height:0px;}
div[data-testid="stDecoration"]{visibility:hidden; height:0px;}

.stApp{
  background:
    radial-gradient(1200px 700px at 18% 12%, rgba(255, 0, 92, 0.18), transparent 60%),
    radial-gradient(900px 600px at 85% 22%, rgba(255, 0, 92, 0.10), transparent 60%),
    linear-gradient(180deg, #06060A 0%, #06060A 100%);
  color:#F3F5FF;
}

.block-container{
  max-width: 1400px;
  padding-top: 1.0rem;
  padding-bottom: 1.0rem;
}

/* Genel kart */
.card{
  background: rgba(255,255,255,0.06);
  border: 1px solid rgba(255, 0, 92, 0.22);
  border-radius: 22px;
  padding: 18px 18px;
  backdrop-filter: blur(10px);
  box-shadow: 0 18px 60px rgba(0,0,0,0.50);
}

.hr{ height:1px; background: rgba(255, 0, 92, 0.18); margin: 12px 0; }
.muted{ color: rgba(243,245,255,0.72); }
.title{ font-size: 2.05rem; font-weight: 950; letter-spacing: -0.6px; }
.sub{ color: rgba(243,245,255,0.80); font-size: 0.98rem; margin-top: 6px; }

/* Chip */
.chips{ display:flex; gap:10px; flex-wrap:wrap; margin-top:12px; }
.chip{
  display:inline-flex;
  align-items:center;
  gap:8px;
  padding: 9px 12px;
  border-radius: 999px;
  background: rgba(0,0,0,0.25);
  border: 1px solid rgba(255, 0, 92, 0.26);
}
.dot{
  width:10px; height:10px; border-radius:999px;
  background: rgba(255,0,92,0.95);
  box-shadow: 0 0 18px rgba(255,0,92,0.55);
  display:inline-block;
  margin-right:10px;
}

/* Başlıklar */
.h3{ font-size: 1.35rem; font-weight: 950; margin: 0; }

/* Streamlit kolon aralığı */
[data-testid="stHorizontalBlock"]{ gap: 22px; }

/* File uploader (beyaz patlama yok) */
div[data-testid="stFileUploader"] section{
  border: 1px dashed rgba(255, 0, 92, 0.28) !important;
  background: rgba(0,0,0,0.22) !important;
  border-radius: 18px !important;
  padding: 14px !important;
}
div[data-testid="stFileUploader"] section *{
  color: rgba(243,245,255,0.88) !important;
}
div[data-testid="stFileUploader"] button{
  border-radius: 14px !important;
  padding: 0.40rem 0.80rem !important;
  font-weight: 950 !important;
  border: 1px solid rgba(255, 0, 92, 0.30) !important;
  background: rgba(255, 0, 92, 0.20) !important;
  color: rgba(255,255,255,0.95) !important;
}
div[data-testid="stFileUploader"] button:hover{
  background: rgba(255, 0, 92, 0.30) !important;
}

/* Ana butonlar */
.stButton>button{
  border-radius: 16px !important;
  padding: 0.55rem 0.95rem !important;
  font-weight: 950 !important;
  border: 1px solid rgba(255, 0, 92, 0.30) !important;
  background: linear-gradient(135deg, rgba(255,0,92,0.75), rgba(120,0,40,0.70)) !important;
  color: rgba(255,255,255,0.96) !important;
  box-shadow: 0 18px 50px rgba(255,0,92,0.12);
}
.stButton>button:hover{ transform: translateY(-1px); }

/* Progress */
div[data-testid="stProgress"] > div{
  height: 9px;
  border-radius: 999px;
  background: rgba(255,255,255,0.08);
}
div[data-testid="stProgress"] > div > div{
  border-radius: 999px;
  background: rgba(255,0,92,0.85) !important;
}

/* Alt not */
.footnote{
  margin-top: 16px;
  padding: 12px 14px;
  border-radius: 16px;
  border: 1px solid rgba(255,0,92,0.20);
  background: rgba(0,0,0,0.22);
  color: rgba(243,245,255,0.78);
  font-size: 0.92rem;
}

/* =========================
   Popover (Ayarlar) Tema Fix
   ========================= */

/* Popover butonu */
button[data-testid="stPopoverButton"]{
  border-radius: 14px !important;
  padding: 0.45rem 0.75rem !important;
  font-weight: 950 !important;
  border: 1px solid rgba(255,0,92,0.30) !important;
  background: rgba(0,0,0,0.25) !important;
  color: rgba(255,255,255,0.95) !important;
}
button[data-testid="stPopoverButton"]:hover{
  background: rgba(255,0,92,0.18) !important;
}

/* Popover iç panel (beyazlığı yok et) */
div[data-testid="stPopoverBody"]{
  background: rgba(0,0,0,0.92) !important;
  border: 1px solid rgba(255,0,92,0.30) !important;
  border-radius: 18px !important;
  box-shadow: 0 24px 70px rgba(0,0,0,0.70) !important;
}
div[data-testid="stPopoverBody"] *{
  color: rgba(243,245,255,0.92) !important;
}

/* Popover içinde Streamlit bazı container'ları beyaz basabiliyor:
   Bunları da zorla koyulaştırıyoruz (ASİL DÜZELTME) */
div[data-testid="stPopoverBody"] > div,
div[data-testid="stPopoverBody"] section,
div[data-testid="stPopoverBody"] .stMarkdown,
div[data-testid="stPopoverBody"] .stMarkdownContainer,
div[data-testid="stPopoverBody"] [data-testid="stMarkdownContainer"],
div[data-testid="stPopoverBody"] [data-testid="stVerticalBlock"],
div[data-testid="stPopoverBody"] [data-testid="stHorizontalBlock"]{
  background: transparent !important;
}

/* Popover içindeki slider/toggle arka planlarını koyulaştır */
div[data-testid="stPopoverBody"] .stSlider,
div[data-testid="stPopoverBody"] .stToggle{
  background: transparent !important;
}
</style>
""",
    unsafe_allow_html=True,
)

# =============================
# Model load
# =============================
@st.cache_resource
def get_model():
    return load_model(MODEL_PATH, device)

try:
    model, classes = get_model()
except Exception:
    st.error("Model yüklenemedi. Önce modeli üret (train) ve yolu kontrol et.")
    st.stop()

def tr_label(label: str) -> str:
    return LABEL_TR.get(label, label)

@torch.no_grad()
def predict_probs(img: Image.Image):
    x = preprocess_pil(img).to(device)
    logits = model(x)
    return torch.softmax(logits, dim=1).squeeze(0).detach().cpu()

def topk_from_probs(probs, k=5):
    vals, idxs = torch.topk(probs, k=min(k, probs.numel()))
    return [(classes[i], float(v)) for v, i in zip(vals.tolist(), idxs.tolist())]

def apply_temperature(probs, T: float):
    p = torch.clamp(probs, 1e-8, 1.0)
    logits = torch.log(p) / max(T, 1e-6)
    return torch.softmax(logits, dim=0)

# =============================
# Header + Settings popover
# =============================
hl, hr = st.columns([0.78, 0.22], vertical_alignment="top")

with hl:
    st.markdown(
        f"""
<div class="card">
  <div class="title"><span class="dot"></span>Sektör Kampüste Proje</div>
  <div class="sub">ResNet18 (Transfer Learning) • Animals-10 • Streamlit</div>

  <div class="chips">
    <span class="chip">👤 <b>{OWNER}</b></span>
    <span class="chip">🖥️ <b>{device_label}</b></span>
    <span class="chip">🧠 <b>Model:</b> {MODEL_PATH}</span>
    <span class="chip">📌 <b>Acc:</b> {METRICS["Accuracy"]:.4f}</span>
    <span class="chip">🎯 <b>Prec:</b> {METRICS["Precision(macro)"]:.4f}</span>
    <span class="chip">📣 <b>Rec:</b> {METRICS["Recall(macro)"]:.4f}</span>
    <span class="chip">✅ <b>F1:</b> {METRICS["F1(macro)"]:.4f}</span>
  </div>
</div>
""",
        unsafe_allow_html=True,
    )

with hr:
    with st.popover("⚙️ Ayarlar", use_container_width=True):
        st.markdown("**Bu ayarlar modeli değiştirmez.** Sadece sonuçların ekranda nasıl gösterileceğini etkiler.")
        st.markdown("- **Top-K:** Kaç tahmin listelensin?\n"
                    "- **Calibration:** Güven yüzdesini daha gerçekçi göstermeye çalışır.\n"
                    "- **Temperature (T):** Calibration açıkken dağılımı yumuşatır (T artarsa olasılıklar yayılır).")
        st.markdown('<div class="hr"></div>', unsafe_allow_html=True)

        c1, c2 = st.columns([0.55, 0.45])
        with c1:
            topk = st.slider("Top-K", 3, 10, 5, 1)
        with c2:
            calib = st.toggle("Calibration (Aç/Kapat)", value=True)

        T = st.slider("Temperature (T)", 1.0, 3.0, 1.6, 0.1, disabled=(not calib))

# =============================
# Main
# =============================
st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)

left, right = st.columns([1.15, 1.0], gap="large")

with left:
    st.markdown('<div class="h3">📤 Görüntü Yükleme</div>', unsafe_allow_html=True)
    st.markdown("<div class='muted' style='margin-top:6px;'>Desteklenen formatlar: <b>JPG • JPEG • PNG</b> (tek görsel)</div>", unsafe_allow_html=True)

    uploaded = st.file_uploader("Görsel yükle", type=["jpg", "jpeg", "png"], label_visibility="collapsed")

    img = None
    if uploaded:
        img = Image.open(uploaded).convert("RGB")
        st.markdown('<div class="hr"></div>', unsafe_allow_html=True)
        st.image(img, use_container_width=True)
    else:
        st.markdown("<div class='muted' style='margin-top:10px;'>Dosya seç veya sürükle-bırak. Önizleme burada görünecek.</div>", unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)

with right:
    st.markdown('<div class="h3">🔎 Tahmin</div>', unsafe_allow_html=True)
    st.markdown("<div class='muted' style='margin-top:6px;'>Tek tıkla sınıflandırma + Top-K olasılık listesi</div>", unsafe_allow_html=True)

    run = st.button("🚀 Tahmin Et", disabled=(img is None), use_container_width=True)

    if run and img is not None:
        probs = predict_probs(img)
        if calib:
            probs = apply_temperature(probs, T)
        preds = topk_from_probs(probs, k=topk)
        st.session_state["preds"] = preds

    preds = st.session_state.get("preds")

    if preds:
        best_label, best_conf = preds[0]

        st.markdown('<div class="hr"></div>', unsafe_allow_html=True)
        st.markdown("#### ✅ Sonuç")
        st.markdown(f"**{tr_label(best_label)}** <span class='muted'>({best_label})</span>", unsafe_allow_html=True)
        st.progress(best_conf)
        st.markdown(f"<div style='font-weight:950; font-size:2.0rem; margin-top:6px;'>{best_conf:.2%}</div>", unsafe_allow_html=True)

        st.markdown('<div class="hr"></div>', unsafe_allow_html=True)
        st.markdown("#### 🧾 Top-K Tahminler")
        for lbl, conf in preds:
            st.markdown(f"**{tr_label(lbl)}** <span class='muted'>({lbl})</span>", unsafe_allow_html=True)
            st.progress(conf)
            st.markdown(f"<div class='muted' style='margin-top:-6px; margin-bottom:10px;'>{conf:.2%}</div>", unsafe_allow_html=True)
    else:
        st.markdown("<div class='muted' style='margin-top:10px;'>Görsel yükleyip “Tahmin Et”e bas.</div>", unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)

# =============================
# Footer note
# =============================
st.markdown(
    """
<div class="footnote">
<b>Not:</b> En iyi doğruluk için tek hayvan/tek nesne, yakın kadraj ve dengeli ışık önerilir.
Calibration açıksa güven yüzdesi daha “gerçekçi” görünür; model ağırlıkları değişmez.
</div>
""",
    unsafe_allow_html=True,
)
