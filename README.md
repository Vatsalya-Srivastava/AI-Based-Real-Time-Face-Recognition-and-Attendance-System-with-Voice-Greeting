# AI-Based Real-Time Face Recognition and Attendance System with Voice Greeting

## Overview

This project is an AI-powered real-time face recognition and attendance system developed using **Python**, **OpenCV**, and **DeepFace**. The system detects faces from a live webcam feed, identifies registered users using deep learning-based facial embeddings, greets recognized users through voice output, and automatically records attendance.

A multi-stage verification process with majority voting is used to improve recognition accuracy and reduce false predictions.

---

## Features

- Real-time face detection using webcam
- AI-based face recognition
- FaceNet512 embedding model
- Multi-stage face verification
- Majority voting for improved accuracy
- Voice greeting for verified users
- Automatic attendance logging
- Unknown person detection
- Easy addition of new users
- Expandable face database
- Offline operation (no internet required)

---

## Technologies Used

### Programming Language
- Python 3.11

### Computer Vision
- OpenCV
- Haar Cascade Classifier

### Artificial Intelligence / Deep Learning
- DeepFace
- FaceNet512

### Face Matching
- Cosine Similarity

### Voice Assistant
- pyttsx3
