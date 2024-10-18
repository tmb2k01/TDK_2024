import numpy as np
from sklearn.metrics import silhouette_score
from copkmeans.cop_kmeans import cop_kmeans
from scipy.stats import ks_2samp
import matplotlib.pyplot as plt
from src.plot_utils import plot_metrics, plot_confidence
from sklearn.cluster import KMeans
from concurrent.futures import ProcessPoolExecutor
import time

from sklearn.metrics import (
    rand_score,
    adjusted_rand_score,
    normalized_mutual_info_score,
)


def ks_distance(x, y):
    _, pvalue = ks_2samp(x, y, method="asymp")
    return pvalue


def calc_distance(segment, base_segment, f):
    """Calculate the distance between a segment and a base segment."""
    return f(segment, base_segment)


class KS_KMeans:
    def __init__(
        self,
        n_clusters=5,
        basepoints_num=50,
        max_iter=300,
        tol=1e-4,
        kmeans="cop-kmeans",
    ):
        self.n_clusters = n_clusters
        self.basepoints_num = basepoints_num
        self.max_iter = max_iter
        self.tol = tol
        self.kmeans = kmeans

        self.distances = None
        self.basepoints_idx = None
        self.labels = None
        self.centers = None
        self.confidences = None

    def _most_distant_point(self, selected_vectors, all_vectors):
        max_distance, most_distant_idx = -1, -1
        for idx, vector in enumerate(all_vectors):
            min_distance = min(
                np.linalg.norm(np.array(vector) - np.array(selected_vector))
                for selected_vector in selected_vectors
            )
            if min_distance > max_distance:
                max_distance, most_distant_idx = min_distance, idx
        return most_distant_idx

    def _calculate_distances(self, f):
        N = len(self.segments)
        self.basepoints_idx = [0]
        self.distances = np.zeros((N, self.basepoints_num))

        for i in range(self.basepoints_num - 1):
            basepoint_idx = self.basepoints_idx[i]

            with ProcessPoolExecutor() as executor:
                futures = {
                    executor.submit(
                        calc_distance, segment, self.segments[basepoint_idx], f
                    ): idx
                    for idx, segment in enumerate(self.segments)
                }

                for future in futures:
                    idx = futures[future]
                    self.distances[idx, i] = future.result()

            new_segment = None

            if len(self.segments) == self.basepoints_num:
                new_segment = i + 1
            else:
                new_segment = self._most_distant_point(
                    self.distances[self.basepoints_idx][:, : i + 1],
                    self.distances[:, : i + 1],
                )
            self.basepoints_idx.append(new_segment)

    def calculate_cop_kmeans(self, n_clusters):
        if self.distances is None:
            raise ValueError("Distances have not been calculated yet.")
        self.labels, self.centers = cop_kmeans(
            self.distances,
            n_clusters,
            cl=self.cannot_link,
            max_iter=self.max_iter,
            tol=self.tol,
        )
        self.calculate_confidence(self.distances, self.centers)

    def calculate_kmeans(self, n_clusters):
        if self.distances is None:
            raise ValueError("Distances have not been calculated yet.")
        kmeans = KMeans(n_clusters=n_clusters, max_iter=self.max_iter, tol=self.tol)
        self.labels = kmeans.fit_predict(self.distances)
        self.centers = kmeans.cluster_centers_
        self.calculate_confidence(self.distances, self.centers)

    def fit(self, segments, f=ks_distance, path=None):
        start_time = time.time()  # Start timing

        self.segments = segments
        self.cannot_link = [(i, i + 1) for i in range(len(segments) - 1)]
        self.basepoints_num = min(self.basepoints_num, len(segments))

        if path is not None:
            self.load_distances(path)
        else:
            self._calculate_distances(f)

        if self.kmeans == "cop-kmeans":
            self.calculate_cop_kmeans(self.n_clusters)
        elif self.kmeans == "kmeans":
            self.calculate_kmeans(self.n_clusters)
        else:
            raise ValueError("Invalid kmeans algorithm")

        end_time = time.time()  # End timing
        duration = end_time - start_time  # Calculate the duration
        print(f"Fit function executed in {duration:.1f} seconds")

        return self

    def predict(self, segments):
        distances = np.zeros((len(segments), self.basepoints_num))
        for i in range(self.basepoints_num - 1):
            basepoint_segment = self.segments[self.basepoints_idx[i]]
            distances[:, i] = [
                ks_distance(segment, basepoint_segment) for segment in segments
            ]

        centers_array = np.array(self.centers)
        labels = np.argmin(
            np.linalg.norm(distances[:, :, np.newaxis] - centers_array.T, axis=1),
            axis=1,
        )
        return labels

    def load_distances(self, path):
        data = np.load(path)
        self.distances = data

    def calculate_confidence(self, points, centroids):
        self.confidence = []
        for point in points:
            centroid_distances = np.linalg.norm(point - centroids, axis=1)
            closest_distances = np.sort(centroid_distances)[:2]
            if closest_distances[1] >= 2 * closest_distances[0]:
                self.confidence.append(1)
            else:
                self.confidence.append(
                    (closest_distances[1] - closest_distances[0]) / closest_distances[0]
                )

    def filter_by_confidence(self, gt_labels, threshold=0.8):
        filtered_predictions = []
        filtered_true_labels = []
        for pred, gt_label, conf in zip(self.labels, gt_labels, self.confidence):
            if conf >= threshold:
                filtered_predictions.append(pred)
                filtered_true_labels.append(gt_label)
        return filtered_predictions, filtered_true_labels

    def _compute_inertia(self, data, centers, clusters):
        data = np.array(data)
        centers = np.array(centers)
        clusters = np.array(clusters)
        squared_distances = np.sum((data - centers[clusters]) ** 2, axis=1)
        inertia = np.sum(squared_distances)
        return inertia

    def calculate_metrics(self, gt_labels):
        ri = rand_score(gt_labels, self.labels)
        ari = adjusted_rand_score(gt_labels, self.labels)
        nmi = normalized_mutual_info_score(gt_labels, self.labels)
        return ri, ari, nmi

    def plot_metrics(self, gt_labels):
        ri, ari, nmi = self.calculate_metrics(gt_labels)
        plot_metrics(ri, ari, nmi)

    def plot_optimal_clusters(self, max_clusters=10):
        if len(self.distances) < max_clusters:
            max_clusters = len(self.distances)

        elbow_scores = []
        silhouette_scores = []

        for k in range(2, max_clusters):
            labels, centers = cop_kmeans(self.distances, k, cl=self.cannot_link)
            inertia = self._compute_inertia(self.distances, centers, labels)
            elbow_scores.append(inertia)
            silhouette_avg = silhouette_score(self.distances, labels)
            silhouette_scores.append(silhouette_avg)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 6))

        ax1.plot(
            range(2, max_clusters), elbow_scores, marker="o", linestyle="-", color="b"
        )
        ax1.set_xlabel("Number of Clusters")
        ax1.set_ylabel("Within-Cluster Sum of Squares (WCSS)")
        ax1.set_title("Elbow Method")

        ax2.plot(
            range(2, max_clusters),
            silhouette_scores,
            marker="o",
            linestyle="-",
            color="b",
        )
        ax2.set_xlabel("Number of Clusters")
        ax2.set_ylabel("Silhouette Score")
        ax2.set_title("Silhouette Analysis")

        plt.show()

    def plot_confidence(self):
        plot_confidence(self.confidence)
