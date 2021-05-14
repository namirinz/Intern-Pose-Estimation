"""Hyperparameter on PCA based on ground truth file."""
import os
import pandas as pd
import pickle

import matplotlib.pyplot as plt

from sklearn.decomposition import PCA


UNSELECTED_COLS = ['filename', 'class_name']

def main():
    current_dir = os.path.dirname(__file__)
    csv_dir = "../data/processed/action/csv/ground_truth.csv"
    pca_weight_dir = "../models/weights/pca.pkl"

    full_csv_dir = os.path.abspath(os.path.join(current_dir, csv_dir))
    full_pca_weight_dir = os.path.abspath(
        os.path.join(current_dir, pca_weight_dir)
    )

    df = pd.read_csv(full_csv_dir)
    df_feature = df.drop(columns=UNSELECTED_COLS)
    
    pca_visual = PCA()
    pca_visual.fit_transform(df_feature.values)
    plt.plot(pca_visual.explained_variance_)
    plt.show()

    num_features = int(input('Enter the number of feature to use on PCA: '))
    pca_usage = PCA(n_components=num_features)
    pca_usage.fit_transform(df_feature.values)

    with open(full_pca_weight_dir, 'wb') as pca_file:
        print(f'Saving PCA with {num_features} n_components.')
        pickle.dump(pca_usage, pca_file)

if __name__ == '__main__':
    main()
