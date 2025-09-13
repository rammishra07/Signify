# SilentVoice - Real-Time Sign Language Translator

![GitHub last commit](https://img.shields.io/github/last-commit/rammishra07/Signify)


## Overview

SigniFy is a real-time sign language translation system that leverages computer vision and deep learning to translate hand gestures captured from a webcam into readable text. It also provides an API server for image-based sign translation.

---

## Features

- Live webcam sign language recognition using MediaPipe and a 1D CNN model
- Custom gesture collection for expanding recognized sign vocabulary
- FastAPI backend with image upload endpoint for sign translation
- Flask server example with CORS enabled for frontend integration
- Easily extensible for new hand gestures and improved accuracy

---

## Project Structure

- `collect_gestures.py` — Capture and label hand landmarks data via webcam
- `train_model.py` — Train a 1D CNN on gesture data for classification
- `real_time_sign_translation.py` — Real-time sign translation webcam app
- `main.py` — FastAPI backend providing `/translate-sign` API endpoint
- `app.py` — Flask server example
- `requirements.txt` — Python dependencies
- `gesture_data.csv` — Collected gesture landmarks data (usually gitignored)
- `sign_language_model.h5` — Saved trained model file (usually gitignored)
- `label_encoder.npy` — Label encoder classes file (usually gitignored)

---

## Setup and Installation

### Prerequisites

- Python 3.8 or higher
- Recommended: virtual environment

### Installation
git clone https://github.com/rammishra07/Signify.git
cd Signify

python -m venv venv

Activate venv:
Windows:
venv\Scripts\activate

macOS/Linux:
source venv/bin/activate

pip install -r requirements.txt


---

## Usage

### 1. Collect Gestures

Run to capture new hand gestures with labels:

python collect_gestures.py

text

### 2. Train Model

Train your CNN model on the collected data:

python train_model.py

text

### 3. Run Real-Time Translator

Start live webcam sign language translation:

python real_time_sign_translation.py

text

### 4. Run FastAPI Server

Start the API serving the translation endpoint:

uvicorn main:app --reload

text

API docs available at: `http://localhost:8000/docs`

---

## Contributing

Contributions are welcome! Feel free to open issues or pull requests to improve functionality, add gestures, or enhance accuracy.

---

## License

This project is licensed under the MIT License.

---

## Contact

For questions or collaboration, please open an issue or contact [rammishra07](https://github.com/rammishra07).


