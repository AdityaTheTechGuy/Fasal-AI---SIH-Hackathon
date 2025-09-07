Fasal AI - Smart Crop Recommendation System 🌾
Fasal AI is an intelligent, user-friendly web application designed to help farmers make informed decisions by providing crop recommendations based on soil and climate data. Powered by a machine learning model, this application leverages data science to bring precision agriculture to your fingertips.

This project was developed as a part of the Smart India Hackathon 2025 initiative, aiming to revolutionize agriculture with Artificial Intelligence.

✨ Features
🤖 AI-Powered Recommendations: Utilizes a trained Scikit-learn model with 98.86% accuracy to predict the most suitable crop.

🌐 Multi-Language Support: Fully internationalized interface supporting English, Hindi, Marathi, Gujarati, and Bangla.

🗣️ Text-to-Speech: An integrated "Listen" feature that reads out on-screen text in the selected language.

📊 Dynamic & Interactive UI: Easy-to-use sliders for inputting soil and climate parameters.

📈 Data Visualization: Presents the top crop recommendations with confidence scores in a clear and understandable format.

🚀 Automated Setup: Includes a simple one-click launcher script (start.bat) for Windows to automatically set up the environment and run the application.

☁️ Ready for Deployment: The application is configured for easy deployment on cloud platforms like Render.

🛠️ Technology Stack
Backend: Python, Flask

Machine Learning: Scikit-learn, NumPy

Internationalization (i18n): Flask-Babel

Text-to-Speech: gTTS (Google Text-to-Speech)

Frontend: HTML, CSS, JavaScript

Deployment: Gunicorn, Render

🚀 Getting Started
Follow these instructions to get a copy of the project up and running on your local machine.

Prerequisites
Python 3: Make sure you have Python 3 installed on your system. You can download it from python.org.

Important: During installation on Windows, ensure you check the box that says "Add Python to PATH".

Installation & Launch
This project includes an automated launcher script to make setup effortless.

On Windows:

Download or clone the repository to your local machine.

Navigate to the project folder (Fasal AI/).

Simply double-click the start.bat file.

That's it! The script will automatically:

Create a virtual environment if it doesn't exist.

Install all the required packages from requirements.txt.

Compile the translation files.

Start the Flask web server.

On macOS / Linux:

Download or clone the repository.

Open your terminal and navigate to the project folder.

Run the following commands:

# Create and activate a virtual environment
python3 -m venv fasal_ai_env_311
source fasal_ai_env_311/bin/activate

# Install dependencies
pip install -r requirements.txt

# Compile translations
pybabel compile -d translations

# Run the app
python app.py

Once the server is running, you can access the application in your web browser at https://www.google.com/search?q=http://127.0.0.1:5000.

🌐 Internationalization (i18n)
The application uses the Flask-Babel library for translations. All translatable text is located in the .po files within the translations directory.

To add or edit translations:

Extract text: Run pybabel extract -F babel.cfg -o messages.pot . to find any new text in the templates.

Update source files: Run pybabel update -i messages.pot -d translations to update the .po files for each language.

Translate: Edit the msgstr fields in each .po file (e.g., translations/hi/LC_MESSAGES/messages.po).

Compile: Run pybabel compile -d translations to compile the changes. The start.bat script runs this automatically.

📂 Project Structure
Fasal AI/
├── static/                 # CSS and JavaScript files
│   ├── app.js
│   └── style.css
├── templates/              # HTML templates
│   └── index.html
├── translations/           # Translation files for each language
│   └── hi/LC_MESSAGES/
│       ├── messages.po
│       └── messages.mo
├── .gitignore              # Files for Git to ignore (like the venv)
├── app.py                  # The main Flask application logic
├── babel.cfg               # Configuration for Flask-Babel
├── crop_recommendation_model.pkl # The trained machine learning model
├── label_encoder.pkl       # The label encoder for the model
├── requirements.txt        # List of Python packages to install
└── start.bat               # Automated launcher for Windows

📜 License
This project is licensed under the MIT License - see the LICENSE.md file for details.