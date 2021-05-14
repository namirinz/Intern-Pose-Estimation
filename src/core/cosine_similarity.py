"""Select answer from cosine similarity with ground truth"""
import os

import numpy as np
import pandas as pd

from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
from src.core.feature_engineer import pca_transform
from typing import Tuple, List


CURRENT_DIR = os.path.dirname(__file__)
CSV_DIR = "../data/processed/action/csv/ground_truth.csv"
FULL_CSV_DIR = os.path.abspath(os.path.join(CURRENT_DIR, CSV_DIR))
DF = pd.read_csv(FULL_CSV_DIR)

def get_most_action(
        classes: List[str],
        n_frequency=2,
    ) -> List[str]:
    """Get the most n_frequency class from sorted classes scores.

    Args:
        classes (List[str]): List of class from top_cosin().
        n_frequency (int): Select most n_frequency action in classes
            default is 2.

    Returns:
        List of classes with n_frquency length.
    """
    unique_class, counts = np.unique(classes, return_counts=True)
    sort_count_value = sorted(list(zip(counts, unique_class)), reverse=True)
    num_select = min(n_frequency, len(sort_count_value))

    return sort_count_value[num_select]


def get_top_cosine(
        cosine_score: np.ndarray,
        num_tops: int
    ) -> List[Tuple[str, float]]:
    """Sort and select top num_tops cosine similarity score.

    Args:
        cosine_score (np.ndarray): Cosine similarity score.
        num_tops (int): Number of top score to be selected.

    Returns:
        List of tuple with class name (str) and cosine similarity score (float).
    """
    df = DF.copy()
    df['cos_score'] = cosine_score
    df_sorted = df.sort_values(by='score')
    top_score = df_sorted.iloc[:num_tops]['cos_score'].values
    top_class = df_sorted.iloc[:num_tops]['class_name'].values

    classes_scores = list(zip(top_score, top_class))

    return classes_scores


def compute_cosine(
        ground_truth: np.ndarray,
        features: np.ndarray,
        pca: PCA,
    ):
    """Compute cosine similarity between ground truth and input.
    
    Args:
        ground_truth (np.ndarrah): [N, 196]
    """
    ground_truth_pca = pca.transform(ground_truth)
    features_pca = pca.transform(features)

    features_pca_expand = np.expand_dims(features_pca, axis=0)

    return cosine_similarity(ground_truth_pca, features_pca_expand)
