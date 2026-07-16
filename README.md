 PawHealth AI

PawHealth AI is an AI-powered web application designed to assist in the early detection of health conditions in stray dogs using computer vision. Users can upload or capture an image of a stray dog, and the system analyzes the image to determine whether the dog appears healthy or infected. If an infection is detected, the application identifies the most probable disease, provides first-aid recommendations, and suggests nearby veterinary clinics and animal welfare organizations.

The application is designed to operate offline after deployment, making it suitable for areas with limited or no internet connectivity.

Features

- AI-powered healthy vs infected classification
- Detection of eight common canine diseases
- Confidence score for predictions
- First-aid and treatment recommendations
- Disease urgency assessment
- Offline NGO and veterinary clinic locator
- GPS and city-based search
- Mobile-friendly interface
- No external APIs required during runtime

Diseases Covered

- Mange
- Ringworm
- Canine Distemper
- Canine Parvovirus
- Tick and Flea Infestation
- Open Wound or Injury
- Ocular Infection
- Bacterial Skin Infection (Pyoderma)

Technology Stack

 Frontend

- HTML5
- CSS3
- JavaScript

Backend

- Flask
- Python

 Machine Learning

- PyTorch
- TorchVision
- MobileNetV3-Small
- NumPy
- Pillow

 Deployment

- Gunicorn

System Architecture

1. User uploads or captures an image.
2. The image is preprocessed.
3. MobileNetV3-Small predicts whether the dog is healthy or infected.
4. If infected, a rule-based disease classification engine identifies the most likely disease.
5. The system generates:
   - Disease prediction
   - Confidence score
   - Urgency level
   - Symptoms
   - First-aid guidance
   - Treatment recommendations
6. Nearby NGOs and veterinary clinics are displayed using the offline location database.



Future Improvements

- Expand NGO coverage to additional cities.
- Develop Android and iOS mobile applications.
- Support real-time disease tracking for individual stray dogs.
- Improve disease classification using a larger multi-class dataset.
- Add multilingual support.

Limitations

- The disease classification is intended as an initial screening tool and is not a substitute for professional veterinary diagnosis.
- NGO information currently covers selected Indian cities.
- Prediction accuracy may vary depending on image quality and lighting conditions.

License

This project is developed for educational and research purposes.

Author

Kanmani Jayakumar

B.Sc. Computer Science with Artificial Intelligence

