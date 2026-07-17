import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
import warnings
warnings.filterwarnings('ignore')

# ============================================
# 1. โหลดข้อมูล (ใช้ header=0 เพราะไฟล์มี header อยู่แล้ว)
# ============================================
print("📊 กำลังโหลดข้อมูล...")

df = pd.read_csv('loan_data.csv', header=0)

print(f"✅ โหลดข้อมูลสำเร็จ: {df.shape[0]} แถว, {df.shape[1]} คอลัมน์")
print("\n📋 ตัวอย่างข้อมูล:")
print(df.head())
print("\n📋 ชื่อคอลัมน์:")
print(df.columns.tolist())

# ============================================
# 2. สำรวจข้อมูล
# ============================================
print("\n📈 ข้อมูลสรุป:")
print(df.describe())
print("\n🔍 ข้อมูลที่ขาดหายไป:")
print(df.isnull().sum())

# ============================================
# 3. เตรียมข้อมูล (Preprocessing)
# ============================================
print("\n🔧 กำลังเตรียมข้อมูล...")

# ตรวจสอบค่าของ loan_status ก่อนแปลง
print("\n🎯 ค่าเริ่มต้นของ loan_status:")
print(df['loan_status'].value_counts())

# แปลง loan_status ให้เป็นตัวเลข (กรณีที่เป็น string '0', '1')
df['loan_status'] = pd.to_numeric(df['loan_status'], errors='coerce')

# ลบแถวที่มีค่า NaN ออก
df = df.dropna()

print("\n🎯 ค่าของ loan_status หลังแปลง:")
print(df['loan_status'].value_counts())

# Gender: male=1, female=0
df['person_gender'] = df['person_gender'].map({'male': 1, 'female': 0})

# Education: แปลงเป็นลำดับชั้น
education_map = {
    'High School': 1,
    'Associate': 2,
    'Bachelor': 3,
    'Master': 4,
    'Doctorate': 5
}
df['person_education'] = df['person_education'].map(education_map)

# Home Ownership
home_ownership_map = {
    'RENT': 0,
    'OWN': 1,
    'MORTGAGE': 2,
    'OTHER': 3
}
df['person_home_ownership'] = df['person_home_ownership'].map(home_ownership_map)

# Loan Intent: ใช้ Label Encoding
le_intent = LabelEncoder()
df['loan_intent'] = le_intent.fit_transform(df['loan_intent'].astype(str))

# Previous Loan Defaults: Yes=1, No=0
df['previous_loan_defaults_on_file'] = df['previous_loan_defaults_on_file'].map({'Yes': 1, 'No': 0})

print("✅ เตรียมข้อมูลสำเร็จ")

# ============================================
# 4. แยก features และ target
# ============================================
X = df.drop('loan_status', axis=1)
y = df['loan_status'].astype(int)

print(f"\n📊 Features: {X.shape}")
print(f"🎯 Target: {y.shape}")
print(f"\n การกระจายของ Target:")
print(y.value_counts())

# ============================================
# 5. แบ่งข้อมูล train/test (80:20)
# ============================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\n📈 Training set: {X_train.shape[0]} samples")
print(f"📉 Test set: {X_test.shape[0]} samples")

# ============================================
# 6. ปรับสเกลข้อมูล
# ============================================
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("✅ ปรับสเกลข้อมูลสำเร็จ")

# ============================================
# 7. สร้างและฝึกโมเดล SVM
# ============================================
print("\n🤖 กำลังสร้างโมเดล SVM...")
svm_model = SVC(
    kernel='rbf',
    C=1.0,
    gamma='scale',
    probability=True,
    random_state=42,
    class_weight='balanced'
)

svm_model.fit(X_train_scaled, y_train)
print("✅ ฝึกโมเดลสำเร็จ")

# ============================================
# 8. ประเมินผลโมเดล
# ============================================
print("\n📊 กำลังประเมินผลโมเดล...")
y_pred = svm_model.predict(X_test_scaled)

accuracy = accuracy_score(y_test, y_pred)
print(f"\n🎯 Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")

print("\n📋 Classification Report:")
print(classification_report(y_test, y_pred))

print("\n🔢 Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# ============================================
# 9. บันทึกโมเดล
# ============================================
print("\n💾 กำลังบันทึกโมเดล...")
joblib.dump(svm_model, 'svm_loan_model.pkl')
joblib.dump(scaler, 'scaler.pkl')
joblib.dump(le_intent, 'label_encoder_intent.pkl')

feature_names = X.columns.tolist()
joblib.dump(feature_names, 'feature_names.pkl')

intent_classes = le_intent.classes_.tolist()
joblib.dump(intent_classes, 'intent_classes.pkl')

print("✅ บันทึกโมเดลสำเร็จ!")
print("\n ไฟล์ที่บันทึก:")
print("  - svm_loan_model.pkl")
print("  - scaler.pkl")
print("  - label_encoder_intent.pkl")
print("  - feature_names.pkl")
print("  - intent_classes.pkl")

print("\n✨ เสร็จสมบูรณ์!")