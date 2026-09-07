import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import seaborn as sns
from sklearn.metrics import confusion_matrix

# Configuracion
PATH_BASE = "./dataset-cazas"
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
MODELO_BINARIO = "cazas_final.keras"
CLASES_FILE = "clases_nombres.npy"

# Cargar datos
print("Cargando dataset...")
train_ds = tf.keras.utils.image_dataset_from_directory(
    PATH_BASE, validation_split=0.2, subset="training", seed=123,
    image_size=IMG_SIZE, batch_size=BATCH_SIZE)
val_ds = tf.keras.utils.image_dataset_from_directory(
    PATH_BASE, validation_split=0.2, subset="validation", seed=123,
    image_size=IMG_SIZE, batch_size=BATCH_SIZE)
class_names = train_ds.class_names
labels = np.load(CLASES_FILE)

AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

# 1. Ejemplo de clases
print("Generando: 01_ejemplo_clases.png")
folders = sorted([f for f in os.listdir(PATH_BASE) if os.path.isdir(os.path.join(PATH_BASE, f))])
fig, axes = plt.subplots(1, len(folders), figsize=(15, 5))
for i, name in enumerate(folders):
    folder_path = os.path.join(PATH_BASE, name)
    files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    if files:
        img = mpimg.imread(os.path.join(folder_path, files[0]))
        axes[i].imshow(img)
        axes[i].set_title(f"Clase: {name}")
        axes[i].axis('off')
plt.tight_layout()
plt.savefig('img/01_ejemplo_clases.png', dpi=150, bbox_inches='tight')
plt.close()

# 2. Matriz de confusion
print("Generando: 04_matriz_confusion.png")
model_ready = tf.keras.models.load_model(MODELO_BINARIO)
y_true, y_pred = [], []
for x, y in val_ds:
    y_true.extend(y.numpy())
    y_pred.extend(np.argmax(model_ready.predict(x, verbose=0), axis=-1))

plt.figure(figsize=(10, 8))
cm = confusion_matrix(y_true, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=labels, yticklabels=labels)
plt.title('Matriz de Confusion: Realidad vs Prediccion')
plt.xlabel('Prediccion de la IA')
plt.ylabel('Realidad (Etiqueta)')
plt.savefig('img/04_matriz_confusion.png', dpi=150, bbox_inches='tight')
plt.close()

# 3. Predicciones
print("Generando: 05_predicciones.png")
plt.figure(figsize=(15, 10))
for images, labels_batch in val_ds.take(1):
    preds = model_ready.predict(images, verbose=0)
    for i in range(10):
        actual = labels[labels_batch[i]]
        pred_idx = np.argmax(preds[i])
        predicho = labels[pred_idx]
        confianza = 100 * np.max(preds[i])
        color = 'green' if actual == predicho else 'red'
        plt.subplot(2, 5, i + 1)
        plt.imshow(images[i].numpy().astype("uint8"))
        plt.title(f"REAL: {actual}\nPRED: {predicho} ({confianza:.1f}%)", color=color)
        plt.axis("off")
plt.tight_layout()
plt.savefig('img/05_predicciones.png', dpi=150, bbox_inches='tight')
plt.close()

print("\nGraficas extraidas exitosamente!")
