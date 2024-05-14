
import os
import numpy as np
from PIL import Image
import tensorflow as tf
import warnings

warnings.filterwarnings("ignore")
# %%
# Ścieżka do katalogu głównego
data_dir = "./clouds_confirm"


# %%
def load_images_and_labels(data_dir):
    images = []
    labels = []
    # przechodzenie przez każdy podkatalog w katalogu głównym
    for folder in os.listdir(data_dir):
        folder_path = os.path.join(data_dir, folder)
        if os.path.isdir(folder_path):
            # przechodzenie przez każdy plik w podkatalogu
            for file in os.listdir(folder_path):
                file_path = os.path.join(folder_path, file)
                # wczytanie obrazu
                image = Image.open(file_path)
                # normalizacja do zakresu 0-1
                image = np.array(image) / 255.0
                images.append(image)
                # Przypisywanie etykiety na podstawie nazwy pliku
                label = "not_cloud" if "not_cloud" in file else "cloud"
                labels.append(label)

    return np.array(images), np.array(labels)


# %%
# Wczytywanie obrazów i etykiet
images, labels = load_images_and_labels(data_dir)
# %%
# Konwersja etyket na format numeryczny
labels = np.array([0 if label == "not_cloud" else 1 for label in labels])
# %%
# Ile mamy wystąpień w każdej z klas?
print(np.unique(labels, return_counts=True))
# %% md
# Podział na dane treningowe, walidacyjne i testowe
# %%
from sklearn.model_selection import train_test_split

# Najpierw dzielimy dane na część treningową i pozostałą
train_images, temp_images, train_labels, temp_labels = train_test_split(
    images, labels, test_size=0.30, random_state=42
)
# Pozostałą część dzielimy na zbiory walidacyjny i testowy
val_images, test_images, val_labels, test_labels = train_test_split(
    temp_images, temp_labels, test_size=0.50, random_state=42
)

# %%
print(f"Liczba obrazów treningowych: {len(train_images)}")
print(f"Liczba obrazów walidacyjnych: {len(val_images)}")
print(f"Liczba obrazów testowych: {len(test_images)}")
# %% md
# Model 1
## Bez balansu klasowego
# %%
from keras.api.models import Sequential
from keras.api.layers import Conv2D, MaxPooling2D, Flatten, Dense

# Budowanie modelu
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(100, 100, 1)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(64, activation='relu'),
    Dense(1, activation='sigmoid')
])
# %%

# Kompilacja modelu
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
# %%

# %%
# Wyświetlenie struktury modelu
model.summary()
