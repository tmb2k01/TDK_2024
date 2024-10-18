import matplotlib.pyplot as plt
import numpy as np


def plot_metrics(ri, ari, nmi):
    metrics = {
        "Rand Index": ri,
        "Adjusted Rand Index": ari,
        "Normalized Mutual Information": nmi,
    }
    for metric_name, metric_value in metrics.items():
        print(f"{metric_name}: {metric_value}")

    plt.figure(figsize=(10, 6))
    plt.bar(metrics.keys(), metrics.values())
    plt.xlabel("Metrics")
    plt.ylabel("Scores")
    plt.title("Clustering Metrics")
    plt.ylim(0, 1)
    plt.show()


def plot_confidence(confidence):
    print("\nConfidence:", confidence)
    plt.hist(confidence, bins=40, edgecolor="black", alpha=0.8)
    plt.title("Confidence score")
    plt.xlabel("Value")
    plt.xlim(0, 1.1)
    plt.ylabel("Frequency")
    plt.show()


def plot_length_confidence_distribution(kmeans):
    segment_lengths = np.array([len(segment) for segment in kmeans.segments])

    segment_lengths = [len(segment) for segment in kmeans.segments]

    cmap = plt.get_cmap("viridis")
    colors = cmap(np.linspace(0, 1, len(set(kmeans.labels))))
    label_to_color = {label: colors[i] for i, label in enumerate(set(kmeans.labels))}

    plt.scatter(
        kmeans.confidence,
        segment_lengths,
        c=[label_to_color[label] for label in kmeans.labels],
    )

    plt.xlabel("Confidence Score")
    plt.ylabel("Segment Length")
    plt.title("Dot Plot of Confidence Score vs. Segment Length")

    for label in set(kmeans.labels):
        plt.scatter([], [], c=label_to_color[label], label=f"Label {label}")
    plt.legend()
    plt.show()
