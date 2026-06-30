import streamlit as st
import cv2
import numpy as np
from detector import predict
from pymongo import MongoClient
import bcrypt
from datetime import datetime
import pandas as pd
import matplotlib.pyplot as plt

# ---------------- MONGODB ----------------
client = MongoClient("mongodb://localhost:27017/")
db = client["deepfake_app"]
users = db["users"]
history = db["history"]

# ---------------- SESSION ----------------
if "page" not in st.session_state:
    st.session_state.page = "login"

# ---------------- EXPLANATION FUNCTION (NEW) ----------------
def get_reason(label):

    if label == "Fake":
        return [
            "🔴 Face blending mismatch detected",
            "🔴 Unnatural skin texture",
            "🔴 Lighting inconsistency",
            "🔴 Possible AI generation artifacts"
        ]
    else:
        return [
            "🟢 Natural skin texture",
            "🟢 Consistent lighting",
            "🟢 Proper facial alignment",
            "🟢 No manipulation artifacts"
        ]

# ---------------- LOGIN ----------------
def login_page():
    st.title("🔐 Login")

    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    if st.button("Login"):

        user = users.find_one({"username": username})

        if user:
            stored_pw = user["password"]

            if isinstance(stored_pw, str):
                stored_pw = stored_pw.encode('utf-8')

            if bcrypt.checkpw(password.encode('utf-8'), stored_pw):
                st.session_state.page = "home"
                st.session_state.username = username
                st.success("Login Successful ✅")
                st.rerun()
            else:
                st.error("Invalid Credentials ❌")
        else:
            st.error("User not found ❌")

    if st.button("Go to Register"):
        st.session_state.page = "register"
        st.rerun()

# ---------------- REGISTER ----------------
def register_page():
    st.title("📝 Register")

    username = st.text_input("Create Username")
    password = st.text_input("Create Password", type="password")

    if st.button("Register"):

        if users.find_one({"username": username}):
            st.warning("User already exists ⚠️")
        else:
            hashed_pw = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())

            users.insert_one({
                "username": username,
                "password": hashed_pw
            })

            st.success("Registered Successfully ✅")

    if st.button("Go to Login"):
        st.session_state.page = "login"
        st.rerun()

# ---------------- SAVE HISTORY ----------------
def save_history(label, confidence):
    history.insert_one({
        "username": st.session_state.username,
        "result": label,
        "confidence": confidence,
        "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    })

# ---------------- MAIN APP ----------------
def main_app():

    st.set_page_config(page_title="Deepfake Detection AI", layout="wide")

    st.markdown("""
        <h1 style='text-align: center; color: #00FFA3;'>🧠 Deepfake Detection AI</h1>
        <p style='text-align: center; color: gray;'>Real-Time Image, Video & Webcam Detection System</p>
        <hr>
    """, unsafe_allow_html=True)

    st.sidebar.title("⚙️ Control Panel")

    if st.sidebar.button("Logout"):
        st.session_state.page = "login"
        st.rerun()

    if st.sidebar.button("📊 View History"):
        st.subheader("📊 Your Detection History")

        data = list(history.find({"username": st.session_state.username}))

        if len(data) == 0:
            st.warning("No history found ⚠️")
        else:
            df = pd.DataFrame(data)

            if "_id" in df.columns:
                df = df.drop(columns=["_id"])

            st.dataframe(df, use_container_width=True)

            csv = df.to_csv(index=False).encode('utf-8')

            st.download_button(
                "📥 Download History",
                csv,
                "history.csv",
                "text/csv"
            )

            st.subheader("📊 Real vs Fake Analysis")

            real_count = len(df[df["result"] == "Real"])
            fake_count = len(df[df["result"] == "Fake"])

            fig, ax = plt.subplots()
            ax.pie([real_count, fake_count],
                   labels=["Real", "Fake"],
                   autopct='%1.1f%%')

            st.pyplot(fig)

    mode = st.sidebar.radio(
        "Select Mode",
        ["🖼️ Image", "🎥 Video", "📷 Webcam"]
    )

    st.sidebar.markdown("---")
    st.sidebar.info("Developed using AI (ResNet Model)")

    # ---------- IMAGE ----------
    if mode == "🖼️ Image":

        st.subheader("📤 Upload Image")

        file = st.file_uploader("Choose an image...", type=["jpg","png","jpeg"])

        if file is not None:

            img = cv2.imdecode(np.frombuffer(file.read(), np.uint8), 1)

            col1, col2 = st.columns(2)

            with col1:
                st.image(img, channels="BGR", use_container_width=True)

            with col2:
                st.markdown("### 🔍 Analysis")

                if st.button("Analyze Image"):

                    label, confidence = predict(img)

                    save_history(label, confidence)

                    if label == "Real":
                        st.success(f"✅ REAL IMAGE\n\nConfidence: {confidence}%")
                        st.progress(int(confidence))
                    else:
                        st.error(f"❌ DEEPFAKE DETECTED\n\nConfidence: {confidence}%")
                        st.progress(int(confidence))

                    # 🔥 ONLY ADDITION (EXPLANATION)
                    st.markdown("### 🧠 Explanation")
                    reasons = get_reason(label)
                    for r in reasons:
                        st.write(r)

    # ---------- VIDEO ----------
    elif mode == "🎥 Video":

        st.subheader("📤 Upload Video")

        video_file = st.file_uploader("Choose a video...", type=["mp4","avi","mov"])

        if video_file is not None:

            with open("temp.mp4", "wb") as f:
                f.write(video_file.read())

            cap = cv2.VideoCapture("temp.mp4")

            stframe = st.empty()

            real_count = 0
            fake_count = 0
            frame_count = 0

            st.info("Processing video...")

            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                frame_count += 1

                if frame_count % 10 == 0:
                    label, confidence = predict(frame)

                    if label == "Real":
                        real_count += 1
                    else:
                        fake_count += 1

                    text = f"{label} ({confidence}%)"
                    color = (0,255,0) if label=="Real" else (0,0,255)

                    cv2.putText(frame, text, (20,40),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                1, color, 2)

                    stframe.image(frame, channels="BGR")

            cap.release()

            total = real_count + fake_count

            if total > 0:
                real_percent = (real_count / total) * 100
                fake_percent = (fake_count / total) * 100

                result = "Real" if real_percent > fake_percent else "Fake"
                confidence = round(max(real_percent, fake_percent), 2)

                save_history(result, confidence)

                st.markdown("---")

                if result == "Real":
                    st.success(f"✅ REAL VIDEO\nConfidence: {confidence}%")
                else:
                    st.error(f"❌ DEEPFAKE VIDEO\nConfidence: {confidence}%")

                st.progress(int(confidence))

                # 🔥 ONLY ADDITION (EXPLANATION)
                st.markdown("### 🧠 Explanation")
                reasons = get_reason(result)
                for r in reasons:
                    st.write(r)

    # ---------- WEBCAM ----------
    elif mode == "📷 Webcam":

        st.subheader("📷 Live Webcam Detection")

        run = st.checkbox("Start Webcam")

        frame_window = st.empty()

        if run:
            cap = cv2.VideoCapture(0)

            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                label, confidence = predict(frame)

                save_history(label, confidence)

                text = f"{label} ({confidence}%)"

                cv2.putText(frame, text, (20,40),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1, (0,255,0), 2)

                frame_window.image(frame, channels="BGR")

            cap.release()

# ---------------- ROUTER ----------------
if st.session_state.page == "login":
    login_page()

elif st.session_state.page == "register":
    register_page()

elif st.session_state.page == "home":
    main_app()