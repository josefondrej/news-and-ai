from typing import List, Tuple

import numpy as np
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt

SHAPE = (28, 28)
ROWS, COLS = SHAPE
GRAYSCALE_RANGE = 255


def load_data(n_datapoints: int = 6000) -> List[Tuple[np.ndarray, int]]:
    data = []

    mnist_train = tfds.load(name="mnist", split="train")

    for mnist_example in mnist_train.take(n_datapoints):  # Only take a single example
        image, label = mnist_example["image"], mnist_example["label"]
        image = image.numpy()[:, :, 0].astype(np.float32) / GRAYSCALE_RANGE
        label = (int)(label)
        if label in [3, 4]:
            data.append((image, label))

    return data


def predict(mask: np.ndarray, image: np.ndarray) -> int:
    score = np.average(image * mask)
    if score > 0:
        return 3
    else:
        return 4

def plot_image(image: np.ndarray):
    plt.figure(figsize=(15, 15))
    plt.imshow(image, cmap="gray_r")
    plt.colorbar()
    ax = plt.axes()

    plt.xticks(np.linspace(1, ROWS - 1, ROWS - 1) - 0.5)
    plt.yticks(np.linspace(1, COLS - 1, COLS - 1) - 0.5)

    ax.xaxis.set_ticklabels([])
    ax.yaxis.set_ticklabels([])

    plt.tick_params(length=0)

    ax.grid()
    plt.show()


def plot_datapoint(datapoint: Tuple[np.ndarray, int]):
    image, label = datapoint
    plt.title(f"Number: {label}")
    plot_image(image)


def evaluate_mask(mask: np.ndarray, data: List[Tuple[np.ndarray, int]]):
    correct_predictions = 0
    for datapoint in data:
        image, label = datapoint
        prediction = predict(mask, image)
        if prediction == label:
            correct_predictions += 1

    return correct_predictions / len(data)


def generate_random_mask(shape: Tuple[int, int] = None):
    if shape is None:
        shape = SHAPE

    mask = np.random.uniform(-1, 1, shape)
    return mask


def cross_masks(mother: np.ndarray, father: np.ndarray) -> np.ndarray:
    selector = np.random.randint(0, 2, SHAPE)
    child = father * selector + mother * (1 - selector)
    return child


def get_random_mask_based_on_scores(mask_pool: List[np.ndarray], scores: List[float]) -> np.ndarray:
    probas = np.array(scores) ** 2
    probas = probas / np.sum(probas)
    index = np.where(np.random.multinomial(1, probas))[0][0]
    return mask_pool[index]


def get_best_mask(mask_pool: List[np.ndarray], scores: List[float]) -> np.ndarray:
    try:
        best_index = np.argmax(np.array(scores))
    except:
        best_index = 0
        print("ERROR: Getting best mask.")
    best_mask = mask_pool[best_index]
    return best_mask


def plot_data_sample(data: List[np.ndarray], predictions: List[int] = None, cols: int = 8):
    total = len(data)
    rows = total // cols
    if rows * cols > total:
        rows += 1
    fig, axes = plt.subplots(nrows=rows, ncols=cols, figsize=(20, 7))
    for i in range(cols * rows):
        if i < len(data):
            subplot = axes[i // cols, i % cols]
            subplot.imshow(data[i], cmap="gray_r")
            subplot.axis("off")
            if predictions is not None:
                subplot.set_title(f"{predictions[i]}")
        else:
            subplot.axis("off")

if __name__ == "__main__":
    data_size = 60000
    mask_pool_size = 10000

    data = load_data(data_size)

    mask_pool = [generate_random_mask() for i in range(mask_pool_size)]
    mask_scores = [evaluate_mask(mask, data) for mask in mask_pool]

    good_masks = [mask for mask, score in zip(mask_pool, mask_scores) if score > 0.9]
    average_mask = np.average(good_masks)

    plot_image(average_mask)

    # best_mask = get_best_mask(mask_pool, mask_scores)
    # best_mask_score = evaluate_mask(best_mask, data)
    #
    # plt.title(f"Best mask, correct percentage: {best_mask_score:.3f}")
    # plot_image(best_mask)
