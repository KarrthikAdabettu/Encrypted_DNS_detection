# Encrypted DNS (DoH) Abuse Detection Without Decryption

Detects malicious DNS-over-HTTPS traffic using ML — without decrypting any packets.

## How To Run

1. Install dependencies:
```
pip install -r requirements.txt
```

2. Run the Flask app:
```
python app.py
```

3. Open browser at: http://localhost:5000

## Dataset Modes

| Mode | Samples | Description |
|---|---|---|
| CIC Original | 49 + 50 | Real benchmark from Canadian Institute for Cybersecurity |
| Dataset 600 | 600 | Extended dataset for better model training |
| Dataset 1000 | 1000 | Larger dataset for improved accuracy |
| Dataset 10K | 10,000 | Maximum robustness, 3000 test flows |
| Custom | Your data | Provide your own Wireshark-captured CSV |

## Custom Dataset

For custom data, provide two CSV files in the same format as the CIC dataset:
- Layer 1: DoH vs Non-DoH classification
- Layer 2: Malicious vs Benign DoH classification

## Project Structure

```
project/
├── app.py                          # Flask backend
├── requirements.txt                # Python dependencies
├── templates/
│   └── index.html                  # Frontend dashboard
├── custom_layer1.csv               # Sample custom dataset (Layer 1)
├── custom_layer2.csv               # Sample custom dataset (Layer 2)
└── Traffic_Data-CSVs/
    ├── Layer1_traffic-DoH_and_NonDoH/
    │   ├── merge_first_layer.csv           # CIC original
    │   ├── merge_first_layer_augmented.csv # 600 samples
    │   ├── merge_first_layer_1000.csv      # 1000 samples
    │   └── merge_first_layer_10k.csv       # 10k samples
    └── Layer2_traffic-Malicious_and_Benign_DoH/
        ├── merge_second_layer.csv           # CIC original
        ├── merge_second_layer_augmented.csv # 600 samples
        ├── merge_second_layer_1000.csv      # 1000 samples
        └── merge_second_layer_10k.csv       # 10k samples
```

## Models Used
- Naive Bayes
- K-Nearest Neighbours (k=5)
- Decision Tree
- Random Forest
- Gradient Boost

## Results
- CIC Original: 93.3% accuracy
- Dataset 600: 98%+ accuracy
- Dataset 1000: 99%+ accuracy  
- Dataset 10K: 100% accuracy
