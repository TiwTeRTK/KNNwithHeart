from sklearn.neighbors import KNeighborsClassifier
import streamlit as st 
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

st.title('การจำแนกข้อมูลด้วยเทคนิค KNN')
#st.image("./img/moew.jpg")

col1, col2 = st.columns(2)

with col1:
   st.header("รอด🫀")
   st.image("./img/Heart1.jpg")

with col2:
   st.header("ไม่รอด💀")
   st.image("./img/Heart2.jpg")

html_7 = """
<div style="background-color:#c5f18a;padding:15px;border-radius:15px 15px 15px 15px;border-style:'solid';border-color:black">
<center><h5>การทำนายโรคหัวใจ</h5></center>
</div>
"""
st.markdown(html_7, unsafe_allow_html=True)
st.markdown("")
st.markdown("")

st.subheader("ข้อมูลส่วนแรก 10 แถว")
dt = pd.read_csv("./data/Heart3.csv")
st.write(dt.head(10))
st.subheader("ข้อมูลส่วนสุดท้าย 10 แถว")
st.write(dt.tail(10))

# สถิติพื้นฐาน
st.subheader("📈 สถิติพื้นฐานของข้อมูล")
st.write(dt.describe())

# การเลือกแสดงกราฟตามฟีเจอร์
st.subheader("📌 เลือกฟีเจอร์เพื่อดูการกระจายข้อมูล")
feature = st.selectbox("เลือกฟีเจอร์", dt.columns[:-1])

# วาดกราฟ boxplot
st.write(f"### 🎯 Boxplot: {feature} แยกตามชนิดของดอกไม้")
fig, ax = plt.subplots()
sns.boxplot(data=dt, x='HeartDisease', y=feature, ax=ax)
st.pyplot(fig)

# วาด pairplot
if st.checkbox("แสดง Pairplot (ใช้เวลาประมวลผลเล็กน้อย)"):
    st.write("### 🌺 Pairplot: การกระจายของข้อมูลทั้งหมด")
    fig2 = sns.pairplot(dt, hue='HeartDisease')
    st.pyplot(fig2)

html_8 = """
<div style="background-color:#6BD5DA;padding:15px;border-radius:15px 15px 15px 15px;border-style:'solid';border-color:black">
<center><h5>ทำนายข้อมูล</h5></center>
</div>
"""
st.markdown(html_8, unsafe_allow_html=True)
st.markdown("")

a1 = st.number_input("กรุณาเลือกข้อมูล a1")
a2 = st.number_input("กรุณาเลือกข้อมูล a2")
a3 = st.number_input("กรุณาเลือกข้อมูล a3")
a4 = st.number_input("กรุณาเลือกข้อมูล a4")
a5 = st.number_input("กรุณาเลือกข้อมูล a5")
a6 = st.number_input("กรุณาเลือกข้อมูล a6")
a7 = st.number_input("กรุณาเลือกข้อมูล a7")
a8 = st.number_input("กรุณาเลือกข้อมูล a8")
a9 = st.number_input("กรุณาเลือกข้อมูล a9")
a10 = st.number_input("กรุณาเลือกข้อมูล a10")
a11 = st.number_input("กรุณาเลือกข้อมูล a11")

if st.button("ทำนายผล"):
    #st.write("ทำนาย")
    #dt = pd.read_csv("./data/iris-3.csv") 
   X = dt.drop('HeartDisease', axis=1)
   y = dt.HeartDisease   

   Knn_model = KNeighborsClassifier(n_neighbors=3)
   Knn_model.fit(X, y)  
    
   x_input = np.array([[a1,a2,a3,a4,a5,a6,a7,a8,a9,a10,a11]])
   st.write(Knn_model.predict(x_input))
   
   out=Knn_model.predict(x_input)

   if out[0] == 1:
    st.image("./img/Heart1.jpg")
   else:
    st.image("./img/Heart2.jpg")
else:
    st.write("ไม่ทำนาย")