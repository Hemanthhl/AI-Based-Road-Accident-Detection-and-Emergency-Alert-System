# 🚦 AI-Based Road Accident Detection and Emergency Alert System

##  Overview

The **AI-Based Road Accident Detection and Emergency Alert System** is a Computer Vision and Deep Learning project designed to detect road accidents automatically from CCTV surveillance videos. The system uses **YOLOv8** for real-time vehicle detection, **Multi-Object Tracking** to monitor vehicle movements, and motion analysis techniques to identify collision events. After confirming an accident, the system classifies its severity as **Minor, Major, or Critical**, determines the accident location, and automatically notifies emergency services through **SMS, Email, and Google Maps navigation links**.

The project aims to reduce emergency response time, improve road safety, and support intelligent traffic monitoring by integrating AI-powered accident detection with automated emergency communication.

---

## Objectives

* Detect road accidents automatically from CCTV surveillance videos.
* Perform real-time vehicle detection using YOLOv8.
* Track multiple vehicles across consecutive frames.
* Detect collisions using motion and tracking information.
* Classify accident severity into Minor, Major, and Critical.
* Identify accident locations using GPS coordinates.
* Send automated SMS and Email notifications to emergency services.
* Store accident records and evidence for future reference.
* Provide a user-friendly interface for monitoring accidents.

---

##  Features

* Real-time Vehicle Detection using YOLOv8
* Multi-Object Vehicle Tracking
* Collision Detection
* Accident Severity Classification
* GPS Location Mapping
* Google Maps Navigation Link Generation
* Automated SMS Alerts
* Automated Email Notifications
* Accident Evidence Storage
* SQLite Database Integration
* Streamlit-based User Interface

---

## Technologies Used

### Programming Language

* Python

### Deep Learning & Computer Vision

* YOLOv8
* OpenCV
* Ultralytics

### Database

* SQLite

### APIs & Services

* Twilio API
* OpenRouteService API
* Google Maps
* SMTP Email Service

### Libraries

* NumPy
* Pandas
* Matplotlib
* Streamlit
* Requests
* Geopy

---

## Project Structure

```text
AI-Based-Road-Accident-Detection-and-Emergency-Alert-System
│
├── README.md
├── Requirements.txt
├── app.py
├── accidents.db
├── Road Accident Detection
│
├── dataset/
│   ├── README.md
│   └── Accident Video167
│
├── Images/
│   ├── Architecture.jpg
│   ├── Tracking.jpg
│   ├── Severity.jpg
│   ├── Accident_image.jpg
│   │
│   ├── Interface/
│   │   ├── Interface1.jpg
│   │   ├── Interface2.jpg
│   │   ├── Interface3.jpg
│   │   ├── Interface4.jpg
│   │   ├── Interface5.jpg
│   │   ├── Interface6.jpg
│   │   └── Interface7.jpg
│   │
│   └── Alerts/
│       ├── Hospital_sms_alert.jpg
│       ├── Police_Station_sms_alert.jpg
│       ├── Hospital_email_alert.jpg
│       ├── Police_Station_email_alert.jpg
│       ├── Controlroom_email_alert.jpg
│       └── Call_Record_alert.aac
│
└── Output/
    ├── Dashboard/
    │   ├── Dashboard1.jpg
    │   └── Dashboard2.jpg
    │
    ├── Sample_Evidence/
    │   ├── frame1_video167.jpg
    │   ├── frame2_video167.jpg
    │   └── frame3_video167.jpg
    │
    └── Severity.jpg
```

---

## Dataset

**Dataset:** Balanced Accident Video Dataset

* 401 traffic surveillance videos
* CCTV-based accident scenarios
* Multiple vehicle categories
* Urban and highway traffic footage
* Used for vehicle detection, tracking, collision analysis, and severity classification

> **Note:** The complete dataset is not included in this repository because of GitHub file size limitations. Please download it separately from the original source.

---

## Installation

1. Clone the repository.

```bash
git clone https://github.com/Hemanthhl/AI-Based-Road-Accident-Detection-and-Emergency-Alert-System.git
```

2. Install the required libraries.

```bash
pip install -r requirements.txt
```

3. Run the Streamlit application.

```bash
streamlit run app.py
```

---

## Results

The proposed system successfully performs real-time vehicle detection, multi-object tracking, collision detection, accident severity classification, GPS-based location mapping, automated emergency alert generation, and accident evidence storage. The integration of AI and automated communication helps reduce emergency response time and improves road safety.

---

## Future Enhancements

* Integration with live CCTV camera feeds
* Deployment on cloud platforms
* Mobile application for emergency monitoring
* Smart traffic signal integration
* Accident hotspot prediction using historical data
* AI-based traffic congestion analysis

---

## Author

**Hemanth Gowda H L**

M.Sc. Data Science
REVA University, Bengaluru

* LinkedIn: *https://www.linkedin.com/in/hemanth-gowdahl*
* GitHub: *https://github.com/Hemanthhl*

---

## License

This project is developed for educational and research purposes.
