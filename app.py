import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# ============================================
# ตั้งค่าหน้าเว็บ
# ============================================
st.set_page_config(
    page_title="Loan Default Prediction",
    page_icon="💰",
    layout="wide"
)

# ============================================
# หัวข้อหลัก
# ============================================
st.title("💰 ระบบทำนายการผิดนัดชำระหนี้")
st.markdown("---")

# ============================================
# โหลดโมเดล
# ============================================
@st.cache_resource
def load_models():
    model = joblib.load('svm_loan_model.pkl')
    scaler = joblib.load('scaler.pkl')
    le_intent = joblib.load('label_encoder_intent.pkl')
    feature_names = joblib.load('feature_names.pkl')
    intent_classes = joblib.load('intent_classes.pkl')
    return model, scaler, le_intent, feature_names, intent_classes

# ตรวจสอบไฟล์โมเดล
required_files = [
    'svm_loan_model.pkl', 'scaler.pkl', 
    'label_encoder_intent.pkl', 'feature_names.pkl',
    'intent_classes.pkl'
]

missing_files = [f for f in required_files if not os.path.exists(f)]

if missing_files:
    st.error(f"❌ ไม่พบไฟล์: {', '.join(missing_files)}")
    st.info("""
    **วิธีแก้ไข:**
    1. เปิด Terminal/Command Prompt
    2. รันคำสั่ง: `python train_model.py`
    3. รอจนเห็นข้อความ "✨ เสร็จสมบูรณ์!"
    4. รีเฟรชหน้านี้
    """)
    st.stop()

try:
    model, scaler, le_intent, feature_names, intent_classes = load_models()
except Exception as e:
    st.error(f"❌ เกิดข้อผิดพลาด: {str(e)}")
    st.stop()

# ============================================
# ฟอร์มกรอกข้อมูล
# ============================================
st.header("📝 กรอกข้อมูลผู้กู้")

col1, col2 = st.columns(2)

with col1:
    st.subheader("👤 ข้อมูลส่วนบุคคล")
    
    person_age = st.number_input("อายุ (ปี)", min_value=18, max_value=100, value=25, step=1)
    
    person_gender = st.selectbox("เพศ", ["male", "female"])
    
    person_education = st.selectbox(
        "ระดับการศึกษา",
        ["High School", "Associate", "Bachelor", "Master", "Doctorate"]
    )
    
    person_income = st.number_input(
        "รายได้ต่อปี (บาท)",
        min_value=0, max_value=10000000, value=50000, step=1000
    )
    
    person_emp_exp = st.number_input(
        "ปีประสบการณ์ทำงาน",
        min_value=0, max_value=50, value=2, step=1
    )
    
    person_home_ownership = st.selectbox(
        "สถานะที่อยู่อาศัย",
        ["RENT", "OWN", "MORTGAGE", "OTHER"]
    )

with col2:
    st.subheader("💳 ข้อมูลสินเชื่อ")
    
    loan_amnt = st.number_input(
        "จำนวนเงินกู้ (บาท)",
        min_value=1000, max_value=100000, value=10000, step=1000
    )
    
    # ใช้ classes จริงจากข้อมูล
    loan_intent = st.selectbox("วัตถุประสงค์การกู้", intent_classes)
    
    loan_int_rate = st.number_input(
        "อัตราดอกเบี้ย (%)",
        min_value=0.0, max_value=30.0, value=10.0, step=0.1
    )
    
    loan_percent_income = st.number_input(
        "สัดส่วนเงินกู้ต่อรายได้",
        min_value=0.0, max_value=1.0, value=0.1, step=0.01
    )
    
    cb_person_cred_hist_length = st.number_input(
        "จำนวนปีประวัติเครดิต",
        min_value=0, max_value=30, value=3, step=1
    )
    
    credit_score = st.number_input(
        "คะแนนเครดิต",
        min_value=300, max_value=850, value=650, step=1
    )

col3, col4 = st.columns(2)
with col3:
    previous_loan_defaults_on_file = st.selectbox(
        "ประวัติการผิดนัดชำระหนี้",
        ["No", "Yes"]
    )

# ============================================
# ปุ่มทำนาย
# ============================================
st.markdown("---")
if st.button("🔮 ทำนายผล", type="primary", use_container_width=True):
    
    # แปลงข้อมูล
    gender_encoded = 1 if person_gender == "male" else 0
    
    education_map = {
        'High School': 1, 'Associate': 2, 'Bachelor': 3,
        'Master': 4, 'Doctorate': 5
    }
    education_encoded = education_map[person_education]
    
    home_ownership_map = {
        'RENT': 0, 'OWN': 1, 'MORTGAGE': 2, 'OTHER': 3
    }
    home_ownership_encoded = home_ownership_map[person_home_ownership]
    
    intent_encoded = le_intent.transform([loan_intent])[0]
    prev_default_encoded = 1 if previous_loan_defaults_on_file == "Yes" else 0
    
    # สร้าง DataFrame
    input_data = pd.DataFrame({
        'person_age': [person_age],
        'person_gender': [gender_encoded],
        'person_education': [education_encoded],
        'person_income': [person_income],
        'person_emp_exp': [person_emp_exp],
        'person_home_ownership': [home_ownership_encoded],
        'loan_amnt': [loan_amnt],
        'loan_intent': [intent_encoded],
        'loan_int_rate': [loan_int_rate],
        'loan_percent_income': [loan_percent_income],
        'cb_person_cred_hist_length': [cb_person_cred_hist_length],
        'credit_score': [credit_score],
        'previous_loan_defaults_on_file': [prev_default_encoded]
    })
    
    # ปรับสเกลและทำนาย
    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)[0]
    probability = model.predict_proba(input_scaled)[0]
    
    # แสดงผล
    st.markdown("---")
    st.header("📊 ผลการทำนาย")
    
    if prediction == 1:
        st.error("❌ มีแนวโน้มที่จะผิดนัดชำระหนี้")
        st.metric("ความเสี่ยง", "สูง")
    else:
        st.success("✅ มีแนวโน้มที่จะชำระหนี้ได้ตามปกติ")
        st.metric("ความเสี่ยง", "ต่ำ")
    
    col5, col6 = st.columns(2)
    with col5:
        st.metric("ความน่าจะเป็นที่จะชำระปกติ", f"{probability[0]*100:.2f}%")
    with col6:
        st.metric("ความน่าจะเป็นที่จะผิดนัด", f"{probability[1]*100:.2f}%")
    
    st.markdown("### 📈 ความน่าจะเป็นของการทำนาย")
    prob_df = pd.DataFrame({
        'สถานะ': ['ชำระปกติ', 'ผิดนัดชำระ'],
        'ความน่าจะเป็น': [probability[0]*100, probability[1]*100]
    })
    st.bar_chart(prob_df.set_index('สถานะ'))

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: gray;'>"
    "Developed with ❤️ using Streamlit and scikit-learn"
    "</div>",
    unsafe_allow_html=True
)