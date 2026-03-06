# Warehouse Bag Counting Dashboard

## Overview

This project implements a warehouse monitoring dashboard that integrates AI-based bag counting using computer vision.

The system detects sacks carried by workers and counts them when they cross a virtual line.

The dashboard also displays IoT monitoring parameters.

## Features

* Real-time sack detection using YOLOv8
* Line-crossing based bag counting
* Live video integration into dashboard
* IoT monitoring section (dummy parameters)

## Technology Stack

* Python
* YOLOv8 (Ultralytics)
* OpenCV
* Streamlit

## Project Structure

dashboard.py – Dashboard UI and video integration
models/best.pt – Trained sack detection model
videos/scenario1.mp4 – Test video feed

## Run the Dashboard

Install dependencies:

pip install -r requirements.txt

Run dashboard:

python -m streamlit run dashboard.py

Then open:

http://localhost:8501

## Author

Shaik Naimullah
