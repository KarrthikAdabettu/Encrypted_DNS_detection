"""
DoH Abuse Detection - Flask Backend
Run: python app.py  ->  open http://localhost:5000
Modes: cic, dataset600, dataset1000, dataset10k, custom
"""
from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

app = Flask(__name__)

PATHS = {
    "cic": {
        "l1": "Traffic_Data-CSVs/Layer1_traffic-DoH_and_NonDoH/merge_first_layer.csv",
        "l2": "Traffic_Data-CSVs/Layer2_traffic-Malicious_and_Benign_DoH/merge_second_layer.csv",
    },
    "smote600": {
        "l1": "Traffic_Data-CSVs/Layer1_traffic-DoH_and_NonDoH/merge_first_layer_augmented.csv",
        "l2": "Traffic_Data-CSVs/Layer2_traffic-Malicious_and_Benign_DoH/merge_second_layer_augmented.csv",
    },
    "smote1000": {
        "l1": "Traffic_Data-CSVs/Layer1_traffic-DoH_and_NonDoH/merge_first_layer_1000.csv",
        "l2": "Traffic_Data-CSVs/Layer2_traffic-Malicious_and_Benign_DoH/merge_second_layer_1000.csv",
    },
    "smote10k": {
        "l1": "Traffic_Data-CSVs/Layer1_traffic-DoH_and_NonDoH/merge_first_layer_10k.csv",
        "l2": "Traffic_Data-CSVs/Layer2_traffic-Malicious_and_Benign_DoH/merge_second_layer_10k.csv",
    },
}

CLASSIFIERS = {
    "Naive Bayes":    GaussianNB(),
    "KNN (k=5)":      KNeighborsClassifier(n_neighbors=5),
    "Decision Tree":  DecisionTreeClassifier(criterion="entropy", random_state=42),
    "Random Forest":  RandomForestClassifier(n_estimators=10, random_state=42),
    "Gradient Boost": GradientBoostingClassifier(n_estimators=20, learning_rate=0.5, random_state=42),
}

def preprocess(df):
    df = df.drop(columns=['SourceIP','DestinationIP','TimeStamp'], errors='ignore').dropna()
    X = df.iloc[:,:-1].select_dtypes(include=[np.number])
    Y = df.iloc[:,-1].astype(str)
    return X, Y

def run_analysis(l1_path, l2_path):
    df1 = pd.read_csv(l1_path)
    df2 = pd.read_csv(l2_path)
    X1, Y1 = preprocess(df1)
    X2, Y2 = preprocess(df2)
    X1_tr, X1_te, y1_tr, y1_te = train_test_split(X1, Y1, test_size=0.3, random_state=42)
    X2_tr, X2_te, y2_tr, y2_te = train_test_split(X2, Y2, test_size=0.3, random_state=42)
    cv_folds = max(2, min(5, len(X1) // 5))

    results = {
        "layer1": {}, "layer2": {}, "best": {},
        "dataset_info": {}, "flow_table": [],
        "feature_stats": [], "feature_importance": []
    }

    results["dataset_info"] = {
        "l1_samples": int(len(df1)), "l2_samples": int(len(df2)),
        "l1_classes": {str(k): int(v) for k, v in df1.iloc[:,-1].value_counts().items()},
        "l2_classes": {str(k): int(v) for k, v in df2.iloc[:,-1].value_counts().items()},
        "features": int(X2.shape[1]), "test_flows": int(len(X2_te)),
    }

    for name, clf in CLASSIFIERS.items():
        clf.fit(X1_tr, y1_tr)
        acc = round(accuracy_score(y1_te, clf.predict(X1_te)) * 100, 1)
        try: cv = round(cross_val_score(clf, X1, Y1, cv=cv_folds).mean() * 100, 1)
        except: cv = acc
        results["layer1"][name] = {"accuracy": acc, "cv": cv}

    best_acc, best_name, best_pred = 0, "", None
    for name, clf in CLASSIFIERS.items():
        clf.fit(X2_tr, y2_tr)
        y_pred = clf.predict(X2_te)
        acc = round(accuracy_score(y2_te, y_pred) * 100, 1)
        try: cv = round(cross_val_score(clf, X2, Y2, cv=cv_folds).mean() * 100, 1)
        except: cv = acc
        results["layer2"][name] = {"accuracy": acc, "cv": cv}
        if acc > best_acc:
            best_acc, best_name, best_pred = acc, name, y_pred

    classes = sorted(list(set(Y2.values)))
    cm = confusion_matrix(y2_te, best_pred, labels=classes).tolist()
    report = classification_report(y2_te, best_pred, output_dict=True)
    def sm(cls, m): return round(report.get(cls, {}).get(m, 0) * 100, 1)

    results["best"] = {
        "name": best_name, "accuracy": best_acc,
        "confusion_matrix": cm, "classes": classes,
        "precision_benign":    sm("Benign",    "precision"),
        "recall_benign":       sm("Benign",    "recall"),
        "precision_malicious": sm("Malicious", "precision"),
        "recall_malicious":    sm("Malicious", "recall"),
        "f1_benign":           sm("Benign",    "f1-score"),
        "f1_malicious":        sm("Malicious", "f1-score"),
    }

    rf = RandomForestClassifier(n_estimators=10, random_state=42)
    rf.fit(X2_tr, y2_tr)
    feat_names = list(X2.columns)
    importances = rf.feature_importances_
    top_idx = np.argsort(importances)[::-1][:10]
    results["feature_importance"] = [
        {"feature": feat_names[i], "importance": round(float(importances[i]) * 100, 1)}
        for i in top_idx
    ]

    best_clf = [c for n, c in CLASSIFIERS.items() if n == best_name][0]
    best_clf.fit(X2_tr, y2_tr)
    all_preds = best_clf.predict(X2_te)
    show_cols = [c for c in ['PacketLengthMean','PacketLengthMode','FlowBytesSent',
                              'ResponseTimeTimeMean','PacketTimeVariance'] if c in X2_te.columns]
    flow_rows = []
    for i in range(len(X2_te)):
        row = {"flow_id": f"Flow_{i+1:04d}"}
        for col in show_cols:
            row[col] = round(float(X2_te.iloc[i][col]), 3)
        row["actual"]    = str(y2_te.iloc[i])
        row["predicted"] = str(all_preds[i])
        row["correct"]   = bool(y2_te.iloc[i] == all_preds[i])
        flow_rows.append(row)

    flow_rows.sort(key=lambda r: (
        0 if r["predicted"] == "Malicious" else 1,
        -r.get("FlowBytesSent", 0)
    ))
    results["flow_table"] = flow_rows

    mal = df2[df2.iloc[:,-1] == "Malicious"]
    ben = df2[df2.iloc[:,-1] == "Benign"]
    for feat in [f for f in ['PacketLengthMean','FlowBytesSent','ResponseTimeTimeMean',
                              'PacketTimeVariance','PacketLengthMode'] if f in df2.columns]:
        b_avg = float(ben[feat].mean())
        m_avg = float(mal[feat].mean())
        diff  = round(((m_avg - b_avg) / (abs(b_avg) + 1e-9)) * 100, 1)
        results["feature_stats"].append({
            "feature": feat, "benign_avg": round(b_avg, 3),
            "malicious_avg": round(m_avg, 3), "diff_pct": diff,
            "direction": "higher" if m_avg > b_avg else "lower"
        })

    return results

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/analyze", methods=["POST"])
def analyze():
    data    = request.json or {}
    dataset = data.get("dataset", "cic")

    if dataset in PATHS:
        l1, l2 = PATHS[dataset]["l1"], PATHS[dataset]["l2"]
        if not os.path.exists(l1) or not os.path.exists(l2):
            return jsonify({"error": f"Dataset files not found for mode: {dataset}. Make sure the CSV files are in the correct folders."}), 400
    elif dataset == "custom":
        l1 = data.get("custom_l1", "")
        l2 = data.get("custom_l2", "")
        if not os.path.exists(l1) or not os.path.exists(l2):
            return jsonify({"error": "Custom dataset files not found. Check your file paths."}), 400
    else:
        return jsonify({"error": "Unknown dataset mode."}), 400

    try:
        return jsonify(run_analysis(l1, l2))
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True, port=5000)
