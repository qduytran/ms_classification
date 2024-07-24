import streamlit as st
import numpy as np
import pickle
from welch import welch_method
from fooof_algorithm import fooof_tool_drop_bandwidth_feature, fooof_tool_drop_periodic_features

# Hàm tải mô hình từ file
def load_model(file_path):
    with open(file_path, 'rb') as file:
        model = pickle.load(file)
    return model

# Tải mô hình từ file nếu bỏ 3 thành phần tuần hoàn (feature selection)
# model = load_model("model.pkl")
# Tải mô hình từ file nếu bỏ thành phần tuần hoàn (feature selection) 
model = load_model("interface//model.pkl")

# Giao diện Streamlit
st.title("Predicting multiple sclerosis using EEG data")

# Tải file .set và .fdt cùng một lúc
uploaded_files = st.file_uploader("Upload .set and .fdt files (same name)", type=["set", "fdt"], accept_multiple_files=True)

# Lưu file và xử lý
if uploaded_files:
    set_file = None
    fdt_file = None
    
    # Phân loại và lưu file
    for uploaded_file in uploaded_files:
        if uploaded_file.name.endswith(".set"):
            set_file = uploaded_file
        elif uploaded_file.name.endswith(".fdt"):
            fdt_file = uploaded_file
    
    if set_file and fdt_file:
        # Lưu file tạm thời
        set_file_path = "temp.set"
        fdt_file_path = "temp.fdt"
        
        with open(set_file_path, "wb") as f:
            f.write(set_file.getbuffer())
        
        with open(fdt_file_path, "wb") as f:
            f.write(fdt_file.getbuffer())
        
        st.write("The files have been uploaded successfully. Processing...")
        
        # Xử lý file và trích xuất đặc trưng
        frequencies, psd = welch_method(set_file_path)
        #features = fooof_tool_drop_periodic_features(frequencies, psd)
        features = fooof_tool_drop_bandwidth_feature(frequencies, psd)
        
        if features is not None:
            if len(features) == 76: #nếu bỏ thành phần tuần hoàn thì ở đây là 38
                input_data = np.array(features).reshape(1, -1)
                prediction = model.predict(input_data)
                if prediction[0] == 1:
                    st.success('This person shows signs of cognitive decline')
                else:
                    st.success('This person showed no signs of cognitive impairment')
            else:
                st.error("Characteristic processing failure. Please check the .set file.")
    else:
        st.error("Please upload both .set and .fdt files at the same time!")