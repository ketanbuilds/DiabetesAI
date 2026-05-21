# 🩺 DiabetesAI — Prediction Web App

> A beautiful, ML-powered web application for diabetes risk prediction

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Flask](https://img.shields.io/badge/Flask-3.0-green)
![ML](https://img.shields.io/badge/ML-scikit--learn-orange)

---

## 🚀 Run Locally (2 Steps)

**Step 1 — Install dependencies:**
```bash
pip install -r requirements.txt
```

**Step 2 — Start the app:**
```bash
python app.py
```

Then open your browser and go to: **http://localhost:5000**

---

## 🌐 Deploy Free on Render

1. Push this folder to GitHub
2. Go to [render.com](https://render.com) → New → Web Service
3. Connect your GitHub repo
4. Set:
   - Build Command: `pip install -r requirements.txt`
   - Start Command: `gunicorn app:app`
5. Click Deploy → Get your live link!

---

## 📁 Files

```
diabetes_webapp/
├── app.py           ← Flask backend + ML prediction
├── index.html       ← Full frontend (form + results + animations)
├── best_model.pkl   ← Trained Decision Tree model
├── scaler.pkl       ← StandardScaler
├── requirements.txt
├── Procfile         ← For Render deployment
└── README.md
```

---

## 👥 Team

- Mohammad Hasnen Mirza — 21100BTCSDSI09480
- Ketan Agrawal — 23100BTCSFBI14620
- Kirtan Jaiswal — 23100BTCSFBI14621

**Shri Vaishnav Institute of Information Technology, Indore · 2025**
