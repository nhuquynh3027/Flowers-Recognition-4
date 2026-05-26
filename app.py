import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
import urllib.request
import os

# Cấu hình trang web ứng dụng
st.set_page_config(page_title="Flower Recognition AI", layout="centered")

st.title("🌸 Ứng Dụng Nhận Diện 5 Loài Hoa")
st.write("Tải lên một bức ảnh hoa bất kỳ để mô hình AI dự đoán loại hoa.")

# =========================================================================
# TỰ ĐỘNG TẢI MODEL TỪ LINK NGOÀI NẾU TRÊN SERVER STREAMLIT CHƯA CÓ FILE
# =========================================================================
@st.cache_resource
def load_my_model():
    model_path = 'flowers_recognition_model.h5'
    
    # Nếu chưa tìm thấy file model trên bộ nhớ Streamlit, tiến hành tải về
    if not os.path.exists(model_path):
        with st.spinner('Đang tải file mô hình AI từ server dự phòng (Khoảng 15-30 giây, chỉ tải lần đầu)...'):
           
            url = "https://www.dropbox.com/scl/fi/47q6tpg4njj1l4opx8ews/flowers_recognition_model.h5?rlkey=ulr0e5mghcomgtmdspxdovf7v&st=p6t06tyu&dl=1" 
            
            # Thực hiện tải file
            urllib.request.urlretrieve(url, model_path)
            st.success('✅ Tải mô hình thành công!')
            
    return tf.keras.models.load_model(model_path, compile=False)

try:
    model = load_my_model()
    class_labels = {0: 'daisy', 1: 'dandelion', 2: 'rose', 3: 'sunflower', 4: 'tulip'}
except Exception as e:
    st.error(f"Lỗi hệ thống khi thiết lập mô hình: {e}")

# =========================================================================
# GIAO DIỆN TẢI ẢNH VÀ XỬ LÝ DỰ ĐOÁN (Giữ nguyên)
# =========================================================================
uploaded_file = st.file_uploader("Chọn một tệp ảnh hoa...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Ảnh bạn đã tải lên', use_container_width=True)
    
    with st.spinner('AI đang phân tích bức ảnh...'):
        img_resized = image.resize((128, 128))
        img_array = np.array(img_resized)
        
        if img_array.shape[-1] != 3:
            img_resized = image.convert("RGB").resize((128, 128))
            img_array = np.array(img_resized)
            
        img_array = img_array / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        
        predictions = model.predict(img_array)
        prediction_idx = np.argmax(predictions)
        
        flower_name = class_labels[prediction_idx]
        confidence = predictions[0][prediction_idx] * 100
        
    st.success(f"### Kết quả dự đoán: **{flower_name.upper()}**")
    st.info(f"Độ tin cậy của mô hình AI: **{confidence:.2f}%**")