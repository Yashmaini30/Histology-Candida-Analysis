#!/usr/bin/env python3
"""
Train an SVM verifier on morphological features and save the calibrated model.
Usage: python train_svm_verifier.py
"""
import argparse
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import classification_report, confusion_matrix


def load_data(csv_path: Path):
    df = pd.read_csv(csv_path)
    # keep only labeled classes
    df = df[df['current_classification'].isin(['yeast', 'pseudohyphae', 'true_hyphae'])].copy()

    # Desired feature order
    desired = ['area', 'perimeter', 'aspect_ratio', 'solidity',
               'extent', 'circularity', 'eccentricity', 'compactness']

    # Provide sensible fallbacks / compute derived features when missing
    # Ensure width/height may be used to compute extent and perimeter approx
    has_width = 'width' in df.columns and 'height' in df.columns

    features = []
    for _, row in df.iterrows():
        f = {}
        f['area'] = float(row['area']) if 'area' in row and not pd.isna(row['area']) else 0.0

        # perimeter: prefer explicit, else approximate with bbox perimeter
        if 'perimeter' in row and not pd.isna(row['perimeter']):
            f['perimeter'] = float(row['perimeter'])
        elif has_width:
            f['perimeter'] = 2.0 * (float(row['width']) + float(row['height']))
        else:
            f['perimeter'] = 0.0

        # aspect_ratio: prefer explicit
        if 'aspect_ratio' in row and not pd.isna(row['aspect_ratio']):
            f['aspect_ratio'] = float(row['aspect_ratio'])
        elif has_width and float(row['height']) > 0:
            a = float(row['width']) / float(row['height'])
            f['aspect_ratio'] = max(a, 1.0 / a)
        else:
            f['aspect_ratio'] = 1.0

        f['solidity'] = float(row['solidity']) if 'solidity' in row and not pd.isna(row['solidity']) else 0.0

        # extent: area / bbox_area if width/height present
        if 'extent' in row and not pd.isna(row['extent']):
            f['extent'] = float(row['extent'])
        elif has_width and float(row['width']) > 0 and float(row['height']) > 0:
            f['extent'] = float(row['area']) / (float(row['width']) * float(row['height']))
        else:
            f['extent'] = 0.0

        f['circularity'] = float(row['circularity']) if 'circularity' in row and not pd.isna(row['circularity']) else 0.0

        # eccentricity: prefer explicit, else approximate from aspect_ratio
        if 'eccentricity' in row and not pd.isna(row['eccentricity']):
            f['eccentricity'] = float(row['eccentricity'])
        else:
            ar = f.get('aspect_ratio', 1.0)
            try:
                minor_over_major = 1.0 / float(ar) if ar > 0 else 1.0
                f['eccentricity'] = float(np.sqrt(max(0.0, 1.0 - minor_over_major ** 2)))
            except Exception:
                f['eccentricity'] = 0.0

        # compactness: perimeter^2 / area if possible
        if 'compactness' in row and not pd.isna(row['compactness']):
            f['compactness'] = float(row['compactness'])
        else:
            a = f['area']
            p = f['perimeter']
            f['compactness'] = float((p ** 2) / a) if a > 0 and p > 0 else 0.0

        features.append([f[k] for k in desired])

    X = np.array(features, dtype=float)
    y = df['current_classification'].values
    return X, y


def main(args):
    csv_path = Path(args.csv)
    out_path = Path(args.out)

    X, y = load_data(csv_path)

    # small transform to area to reduce skew
    X = X.copy()
    X[:, 0] = np.log1p(X[:, 0])

    base_svc = SVC(kernel='rbf', probability=True, class_weight='balanced', random_state=42)
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('svc', base_svc)
    ])

    param_grid = {
        'svc__C': [0.1, 1, 10],
        'svc__gamma': ['scale', 0.01, 0.001]
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    gs = GridSearchCV(pipeline, param_grid, cv=cv, scoring='f1_macro', n_jobs=-1)
    print('Starting grid search...')
    gs.fit(X, y)
    print('Grid search done. Best params:', gs.best_params_)

    # calibrate probabilities
    calibrated = CalibratedClassifierCV(gs.best_estimator_, cv='prefit', method='sigmoid')
    calibrated.fit(X, y)

    y_pred = calibrated.predict(X)
    print('Training classification report:')
    print(classification_report(y, y_pred))
    print('Confusion matrix:')
    print(confusion_matrix(y, y_pred))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(calibrated, out_path)
    print(f'Saved calibrated SVM verifier to {out_path}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv', type=str, default=r'c:\Academics\AI and Ml\Candida\results\morphological_features_analysis.csv',
                        help='Path to feature CSV')
    parser.add_argument('--out', type=str, default=r'c:\Academics\AI and Ml\Candida\src\classification\svm_verifier.pkl',
                        help='Path to write trained verifier')
    args = parser.parse_args()
    main(args)
