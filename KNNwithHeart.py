from sklearn.neighbors import KNeighborsClassifier
import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

st.title('การทำนายข้อมูลโรคหัวใจเทคนิค KNN')

col1, col2 = st.columns(2)

with col1:
    st.header("Heart1")
    st.image("./img/Heart1.jpg")

with col2:
    st.header("Heart2")
    st.image("./img/Heart2.jpg")

html_7 = """
<div style="background-color:#33beff;padding:15px;border-radius:15px 15px 15px 15px;border-style:'solid';border-color:black">
<center><h5>ข้อมูล iris หรือข้อมูลดอกไม้สำหรับทำนาย</h5></center>
</div>
"""
st.markdown(html_7, unsafe_allow_html=True)

st.subheader("ข้อมูลส่วนแรก 10 แถว")
dt = pd.read_csv("./data/Heart3.csv")
st.write(dt.head(10))

st.subheader("ข้อมูลส่วนสุดท้าย 10 แถว")
st.write(dt.tail(10))

st.subheader("📈 สถิติพื้นฐานของข้อมูล")
st.write(dt.describe())

st.subheader("📌 เลือกฟีเจอร์เพื่อดูการกระจายข้อมูล")
feature = st.selectbox("เลือกฟีเจอร์", dt.columns[:-1])

# วาด boxplot
st.write(f"### 🎯 Boxplot: {feature} แยกตามหัวใจ")
fig, ax = plt.subplots()
sns.boxplot(data=dt, x='HeartDiseasey', y=feature, ax=ax)
st.pyplot(fig)

# วาด pairplot
if st.checkbox("แสดง Pairplot (ใช้เวลาประมวลผลเล็กน้อย)"):
    st.write("### 🌺 Pairplot: การกระจายของข้อมูลทั้งหมด")
    fig2 = sns.pairplot(dt, hue='variety')
    st.pyplot(fig2)

html_8 = """
<div style="background-color:#6BD5DA;padding:15px;border-radius:15px 15px 15px 15px;border-style:solid;border-color:black">
<center><h5>ทำนายข้อมูล</h5></center>
</div>
"""
st.markdown(html_8, unsafe_allow_html=True)

# อินพุตจากผู้ใช้
A1 = st.number_input("กรุณาเลือกข้อมูล A1")
A2 = st.number_input("กรุณาเลือกข้อมูล A2")
A3 = st.number_input("กรุณาเลือกข้อมูล A3")
A4 = st.number_input("กรุณาเลือกข้อมูล A4")
A5 = st.number_input("กรุณาเลือกข้อมูล A5")
A6 = st.number_input("กรุณาเลือกข้อมูล A6")
A7 = st.number_input("กรุณาเลือกข้อมูล A7")
A8 = st.number_input("กรุณาเลือกข้อมูล A8")
A9 = st.number_input("กรุณาเลือกข้อมูล A9")
A10 = st.number_input("กรุณาเลือกข้อมูล A10")
A11 = st.number_input("กรุณาเลือกข้อมูล A11")

if st.button("ทำนายผล"):
    # โหลดข้อมูลโรคหัวใจ
    dt = pd.read_csv("./data/Heart3.csv")
    X = dt.drop('HeartDiseasey', axis=1)
    y = dt['HeartDiseasey']

    # สร้างและฝึกโมเดล
    Knn_model = KNeighborsClassifier(n_neighbors=3)
    Knn_model.fit(X, y)

    # เตรียมข้อมูลอินพุต
    x_input = np.array([[A1, A2, A3, A4, A5, A6, A7, A8, A9, A10, A11]])
    prediction = Knn_model.predict(x_input)

    # แสดงผลการทำนาย
    st.subheader("ผลการทำนาย:")
    if prediction[0] == 1:
        st.success("มีแนวโน้มเป็นโรคหัวใจ")
        st.image("./img/Heart1.jpg")
    else:
        st.success("ไม่มีแนวโน้มเป็นโรคหัวใจ")
        st.image("./img/Heart2.jpg")
else:
    st.write("กรุณากรอกข้อมูลให้ครบ แล้วกดปุ่ม 'ทำนายผล'")
