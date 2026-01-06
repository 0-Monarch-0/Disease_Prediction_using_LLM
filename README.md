# 🏥 Disease Prediction System (Capstone Project)

A full-stack Machine Learning–based Disease Prediction System developed as a B.Tech capstone project.  
The system predicts diseases based on user inputs using trained ML models and provides an interactive web interface.

---

## 🚀 Project Overview

This project integrates Machine Learning, Backend APIs, and a Frontend UI into a single end-to-end application.

### Key Features
- Disease prediction using trained ML models
- Cleaned and preprocessed healthcare datasets
- Python-based backend API for inference
- Modern React + Tailwind frontend
- Modular and scalable project structure

---

## 🧠 Machine Learning Details

- Algorithms: Traditional ML classifiers and regression models
- Libraries: Scikit-learn, Pandas, NumPy
- Data Processing: Cleaning, feature selection, encoding
- Model Storage: Pickle (.pkl) files
- Input: Structured health indicators
- Output: Predicted disease or risk estimation

---

## 🛠️ Tech Stack

### Frontend
- React (Vite)
- Tailwind CSS
- JavaScript
- HTML & CSS

### Backend
- Python
- FastAPI / Flask
- REST APIs

### Machine Learning
- Scikit-learn
- Pandas
- NumPy

### Tools
- Git & GitHub
- VS Code
- Node.js
- npm

---

## 📂 Project Structure
disease-predictor/
│
├── api/
│ ├── main.py
│ ├── cleaned_disease_data.csv
│ └── start.bat
│
├── frontend/
│ ├── src/
│ ├── public/
│ ├── package.json
│ └── vite.config.js
│
├── model-training/
│ ├── train.py
│ ├── classifying_with_finaldata.py
│ ├── disease_model.pkl
│ ├── unified_reg_model.pkl
│ └── model_columns.pkl
│
├── cleaned_disease_data.csv
├── Final_data.csv
├── Disease Prediction System Flowchart.png
├── Capstone.pptx
└── README.md


---

## ⚙️ How to Run the Project Locally

### 1. Clone the Repository

git clone https://github.com/your-username/disease-prediction-system.git

cd disease-prediction-system

---

### 2. Backend Setup (Python API)

cd api
pip install -r requirements.txt
python main.py


Backend runs at:
http://127.0.0.1:8000


---

### 3. Frontend Setup (React)

cd frontend
npm install
npm run dev

Frontend runs at:
http://localhost:5173


---

## 📊 Dataset Information

- Cleaned healthcare datasets in CSV format
- Feature engineered for ML training
- Handles missing values, encoding, and normalization

---

## 🧪 Model Training

To retrain the models:

cd model-training
python train.py


Models are saved as .pkl files and loaded by the backend API.

---

## 📈 System Flow

1. User enters health-related inputs
2. Frontend sends request to backend API
3. Backend loads trained ML model
4. Model predicts disease outcome
5. Result is returned to frontend UI

---

## 🎯 Learning Outcomes

- End-to-end ML project development
- Model training and persistence
- REST API development in Python
- Frontend-backend integration
- Real-world project structuring
- GitHub project management

---

## 👨‍💻 Author

Akash Chinnola  
B.Tech Final Year Student  
AI / Machine Learning / Full Stack Development  

This project was developed as part of an academic capstone.

---

## 📌 Future Enhancements

- Deploy backend using Render or Railway
- Deploy frontend using Vercel or Netlify
- Add authentication and authorization
- Improve model accuracy
- Add analytics dashboard
- Expand disease categories

---

⭐ If you found this project interesting, feel free to star the repository.
