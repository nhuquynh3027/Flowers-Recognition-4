import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
import urllib.request
import os

# =====================================================================
# CẤU HÌNH TRANG
# =====================================================================
st.set_page_config(
    page_title="Flower AI · Nhận Diện Hoa",
    page_icon="🌿",
    layout="centered"
)

# =====================================================================
# CSS THIẾT KẾ LẠI GIAO DIỆN
# =====================================================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:ital,wght@0,400;0,500;1,400&family=DM+Sans:wght@300;400;500&display=swap');

/* Nền trang */
html, body, [data-testid="stAppViewContainer"] {
    background: #0d1a12 !important;
    color: #e8f0e4;
}
[data-testid="stHeader"] { background: transparent !important; }
[data-testid="stSidebar"] { display: none; }
.block-container { padding: 2.5rem 1.5rem !important; max-width: 640px !important; }

/* Font chung */
* { font-family: 'DM Sans', sans-serif; }

/* Tiêu đề trang */
.hero-eyebrow {
    font-size: 11px;
    letter-spacing: 0.18em;
    text-transform: uppercase;
    color: #5dbe8a;
    font-weight: 500;
    text-align: center;
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 10px;
    margin-bottom: 0.5rem;
}
.hero-eyebrow::before, .hero-eyebrow::after {
    content: '';
    width: 28px;
    height: 1px;
    background: #5dbe8a;
    opacity: 0.45;
    display: inline-block;
}
.hero-title {
    font-family: 'Playfair Display', serif;
    font-size: 2.3rem;
    font-weight: 400;
    color: #f0f7ec;
    text-align: center;
    line-height: 1.2;
    margin-bottom: 0.5rem;
}
.hero-title em {
    font-style: italic;
    color: #7de0a4;
}
.hero-sub {
    font-size: 13px;
    color: #7a9e85;
    font-weight: 300;
    text-align: center;
    line-height: 1.7;
    margin-bottom: 2rem;
}

/* Vùng upload ảnh */
[data-testid="stFileUploader"] {
    background: rgba(255,255,255,0.03) !important;
    border: 1.5px dashed rgba(93,190,138,0.35) !important;
    border-radius: 16px !important;
    padding: 1.5rem !important;
    transition: border-color 0.25s;
}
[data-testid="stFileUploader"]:hover {
    border-color: rgba(93,190,138,0.65) !important;
}
[data-testid="stFileUploader"] label {
    color: #c8e6ce !important;
    font-size: 14px !important;
    font-weight: 500 !important;
}
[data-testid="stFileUploader"] small {
    color: #5a7a62 !important;
    font-size: 12px !important;
}
[data-testid="stFileUploadDropzone"] button {
    background: #5dbe8a !important;
    color: #0d1a12 !important;
    border: none !important;
    border-radius: 100px !important;
    font-weight: 500 !important;
    font-size: 13px !important;
    padding: 8px 22px !important;
}

/* Ảnh tải lên */
[data-testid="stImage"] img {
    border-radius: 14px !important;
    border: 1.5px solid rgba(93,190,138,0.22) !important;
}

/* Thẻ kết quả */
.result-card {
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(93,190,138,0.22);
    border-radius: 16px;
    padding: 1.5rem;
    margin: 1.25rem 0;
}
.result-eyebrow {
    font-size: 11px;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: #5dbe8a;
    font-weight: 500;
    margin-bottom: 6px;
}
.result-name {
    font-family: 'Playfair Display', serif;
    font-size: 2.2rem;
    font-weight: 500;
    color: #f0f7ec;
    line-height: 1.1;
    text-transform: capitalize;
    margin-bottom: 1.2rem;
}
.conf-row {
    display: flex;
    justify-content: space-between;
    font-size: 12px;
    color: #7a9e85;
    margin-bottom: 8px;
}
.conf-val { color: #5dbe8a; font-weight: 500; }
.conf-bar-bg {
    background: rgba(255,255,255,0.08);
    border-radius: 100px;
    height: 4px;
    overflow: hidden;
}
.conf-bar-fill {
    height: 100%;
    background: linear-gradient(90deg, #5dbe8a, #81e0b0);
    border-radius: 100px;
    transition: width 0.8s ease;
}

/* Divider */
.section-divider {
    display: flex;
    align-items: center;
    gap: 10px;
    margin: 1.5rem 0 1rem;
    font-size: 11px;
    color: #3d5c44;
    text-transform: uppercase;
    letter-spacing: 0.1em;
}
.section-divider::before, .section-divider::after {
    content: '';
    flex: 1;
    height: 1px;
    background: rgba(255,255,255,0.06);
}

/* Lưới loài hoa */
.flowers-grid {
    display: grid;
    grid-template-columns: repeat(5, 1fr);
    gap: 8px;
    margin-bottom: 1rem;
}
.flower-chip {
    padding: 10px 6px;
    border-radius: 10px;
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(255,255,255,0.07);
    text-align: center;
    font-size: 11px;
    color: #7a9e85;
}
.flower-chip .chip-emoji { font-size: 20px; display: block; margin-bottom: 4px; }

/* Spinner / thông báo */
[data-testid="stSpinner"] p { color: #7a9e85 !important; }

/* Streamlit mặc định override */
h1, h2, h3 { color: #f0f7ec !important; }
p, li { color: #c8e6ce; }
.stAlert { border-radius: 12px !important; }
</style>
""", unsafe_allow_html=True)

# =====================================================================
# PHẦN TIÊU ĐỀ TRANG
# =====================================================================
st.markdown("""
<div class="hero-eyebrow">Nhận diện hoa bằng AI</div>
<h1 class="hero-title">Khám phá thế giới <em>hoa tươi</em></h1>
<p class="hero-sub">Tải lên một bức ảnh hoa bất kỳ —<br>mô hình AI sẽ nhận diện và cho bạn kết quả ngay lập tức.</p>
""", unsafe_allow_html=True)

# =====================================================================
# TẢI MÔ HÌNH
# =====================================================================
@st.cache_resource
def load_my_model():
    model_path = 'flowers_recognition_model.h5'
    if not os.path.exists(model_path):
        with st.spinner('Đang tải mô hình AI lần đầu (15–30 giây)…'):
            url = "https://www.dropbox.com/scl/fi/13eqcmw3idsoyoqv84hdg/flowers_recognition_model.h5?rlkey=xiiglckvv690v5uxbutsrt2oc&st=8cjyf1hk&dl=1"
            urllib.request.urlretrieve(url, model_path)
    return tf.keras.models.load_model(model_path, compile=False)

try:
    model = load_my_model()
    class_labels = {0: 'Daisy', 1: 'Dandelion', 2: 'Rose', 3: 'Sunflower', 4: 'Tulip'}
    class_emoji  = {0: '🌼',    1: '🌿',        2: '🌹',   3: '🌻',       4: '🌷'}
except Exception as e:
    st.error(f"Lỗi khi tải mô hình: {e}")
    st.stop()

# =====================================================================
# VÙNG TẢI ẢNH
# =====================================================================
uploaded_file = st.file_uploader(
    "Kéo thả ảnh hoa vào đây hoặc nhấn để chọn file",
    type=["jpg", "jpeg", "png"],
    label_visibility="visible"
)

# =====================================================================
# XỬ LÝ & HIỂN THỊ KẾT QUẢ
# =====================================================================
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Ảnh bạn đã tải lên", use_container_width=True)

    with st.spinner('AI đang phân tích bức ảnh…'):
        img_resized = image.convert("RGB").resize((128, 128))
        img_array  = np.array(img_resized) / 255.0
        img_array  = np.expand_dims(img_array, axis=0)

        predictions    = model.predict(img_array)
        prediction_idx = int(np.argmax(predictions))
        flower_name    = class_labels[prediction_idx]
        flower_emoji   = class_emoji[prediction_idx]
        confidence     = float(predictions[0][prediction_idx]) * 100

    # Thẻ kết quả
    st.markdown(f"""
    <div class="result-card">
        <div class="result-eyebrow">Kết quả nhận diện</div>
        <div class="result-name">{flower_emoji} {flower_name}</div>
        <div class="conf-row">
            <span>Độ tin cậy của mô hình</span>
            <span class="conf-val">{confidence:.1f}%</span>
        </div>
        <div class="conf-bar-bg">
            <div class="conf-bar-fill" style="width:{confidence:.1f}%"></div>
        </div>
    </div>
    """, unsafe_allow_html=True)

# =====================================================================
# LƯỚI 5 LOÀI HOA HỖ TRỢ
# =====================================================================
st.markdown('<div class="section-divider">5 loài hoa được hỗ trợ</div>', unsafe_allow_html=True)

chips_html = '<div class="flowers-grid">'
for i, (name, emoji) in enumerate(zip(class_labels.values(), class_emoji.values())):
    chips_html += f'<div class="flower-chip"><span class="chip-emoji">{emoji}</span>{name}</div>'
chips_html += '</div>'
st.markdown(chips_html, unsafe_allow_html=True)