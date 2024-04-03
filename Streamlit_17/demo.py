import streamlit as st
import pandas as pd
st.write("Welcome to Streamlit Demo Session")

name = st.text_input("Enter your Name")

st.button("Enter")

st.write(f"Hello, {name} Welcome!!")

course = st.radio("Choose your course name",options= ["Data Science", "Data Analysis"])

tenure = st.number_input("Enter your course tenure", min_value=3, max_value=9, format= "%d")

if course == "Data Science":
    st.write(f"you have opted for {course} with tenure {tenure} months")
else:
    st.write(f"you have opted for {course} with tenure {tenure} months")

students = ["ajay", "Raghu", "Swetha", "Priya", "Laxman"]
marks = [100,80,100,100,90]

df = pd.DataFrame({"students":students, "marks":marks})

st.dataframe(df)
st.line_chart(df["marks"])

st.slider("choose your ratings between 0 to 5", max_value=5, step=1)
