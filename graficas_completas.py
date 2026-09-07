import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras import layers, models

# Configuracion
PATH_BASE = "./dataset-cazas"
PATH_TEST = "./dataset-test"
MODELO_BINARIO = "cazas_final.keras"
CLASES_FILE = "clases_nombres.npy"
IMG_SIZE = (224, 224)
BATCH_SIZE = 32

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

# Cargar modelo
print("Cargando modelo...")
model_ready = tf.keras.models.load_model(MODELO_BINARIO)

# ============================================
# 1. COMPORTAMIENTO DEL ENTRENAMIENTO
# ============================================
print("1. Generando graficas de entrenamiento...")

# Simular historial (usando datos reales del notebook)
epochs = list(range(1, 11))

# Datos reales del entrenamiento normal
acc_train = [0.2359, 0.2764, 0.3175, 0.3562, 0.3991, 0.4191, 0.4634, 0.4673, 0.5074, 0.5292]
acc_val = [0.2602, 0.3474, 0.3896, 0.4219, 0.4557, 0.4965, 0.5288, 0.5527, 0.5668, 0.5879]
loss_train = [2.3619, 2.0904, 1.9426, 1.8000, 1.6711, 1.6031, 1.4733, 1.4482, 1.3468, 1.3183]
loss_val = [1.6412, 1.5735, 1.5142, 1.4565, 1.3969, 1.3339, 1.2757, 1.2194, 1.1739, 1.1296]

# Datos del fine-tuning
acc_train_ft = [0.3924, 0.4248, 0.4838, 0.5158, 0.5608, 0.5925, 0.6322, 0.6624, 0.6966, 0.7117]
acc_val_ft = [0.5696, 0.5767, 0.5851, 0.6076, 0.6301, 0.6554, 0.6737, 0.6892, 0.7032, 0.7187]
loss_train_ft = [1.6979, 1.5378, 1.3887, 1.2543, 1.1608, 1.0725, 0.9978, 0.9030, 0.8434, 0.7886]
loss_val_ft = [1.1847, 1.1403, 1.0869, 1.0236, 0.9717, 0.9215, 0.8786, 0.8376, 0.7997, 0.7612]

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Comportamiento del Entrenamiento', fontsize=16, fontweight='bold')

# Entrenamiento normal - Accuracy
axes[0, 0].plot(epochs, [x*100 for x in acc_train], 'bo-', label='Train', linewidth=2, markersize=6)
axes[0, 0].plot(epochs, [x*100 for x in acc_val], 'ro-', label='Validation', linewidth=2, markersize=6)
axes[0, 0].set_title('Fase 1: Entrenamiento Normal', fontsize=12, fontweight='bold')
axes[0, 0].set_xlabel('Epoca')
axes[0, 0].set_ylabel('Accuracy (%)')
axes[0, 0].set_ylim(0, 100)
axes[0, 0].legend(loc='lower right')
axes[0, 0].grid(True, alpha=0.3)

# Entrenamiento normal - Loss
axes[0, 1].plot(epochs, loss_train, 'bo-', label='Train', linewidth=2, markersize=6)
axes[0, 1].plot(epochs, loss_val, 'ro-', label='Validation', linewidth=2, markersize=6)
axes[0, 1].set_title('Fase 1: Entrenamiento Normal', fontsize=12, fontweight='bold')
axes[0, 1].set_xlabel('Epoca')
axes[0, 1].set_ylabel('Loss')
axes[0, 1].legend(loc='upper right')
axes[0, 1].grid(True, alpha=0.3)

# Fine-tuning - Accuracy
axes[1, 0].plot(epochs, [x*100 for x in acc_train_ft], 'bo-', label='Train', linewidth=2, markersize=6)
axes[1, 0].plot(epochs, [x*100 for x in acc_val_ft], 'ro-', label='Validation', linewidth=2, markersize=6)
axes[1, 0].set_title('Fase 2: Fine-Tuning', fontsize=12, fontweight='bold')
axes[1, 0].set_xlabel('Epoca')
axes[1, 0].set_ylabel('Accuracy (%)')
axes[1, 0].set_ylim(0, 100)
axes[1, 0].legend(loc='lower right')
axes[1, 0].grid(True, alpha=0.3)

# Fine-tuning - Loss
axes[1, 1].plot(epochs, loss_train_ft, 'bo-', label='Train', linewidth=2, markersize=6)
axes[1, 1].plot(epochs, loss_val_ft, 'ro-', label='Validation', linewidth=2, markersize=6)
axes[1, 1].set_title('Fase 2: Fine-Tuning', fontsize=12, fontweight='bold')
axes[1, 1].set_xlabel('Epoca')
axes[1, 1].set_ylabel('Loss')
axes[1, 1].legend(loc='upper right')
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('img/09_comportamiento_entrenamiento.png', dpi=150, bbox_inches='tight')
plt.close()
print("   Guardada: img/09_comportamiento_entrenamiento.png")

# ============================================
# 2. REPORTE DE CLASIFICACION DETALLADO
# ============================================
print("2. Generando reporte de clasificacion...")
y_true, y_pred = [], []
for x, y in val_ds:
    y_true.extend(y.numpy())
    y_pred.extend(np.argmax(model_ready.predict(x, verbose=0), axis=-1))

report = classification_report(y_true, y_pred, target_names=labels, output_dict=True)

# Crear tabla visual del reporte
fig, ax = plt.subplots(figsize=(10, 6))
ax.axis('off')

metrics = ['precision', 'recall', 'f1-score', 'support']
data = []
for clase in labels:
    row = [clase]
    for metric in metrics:
        val = report[clase][metric]
        if metric == 'support':
            row.append(f'{val:.0f}')
        else:
            row.append(f'{val:.2f}')
    data.append(row)

# Agregar promedios
data.append(['accuracy', f'{report["accuracy"]:.2f}', '', '', f'{report["macro avg"]["support"]:.0f}'])
data.append(['macro avg', f'{report["macro avg"]["precision"]:.2f}', 
             f'{report["macro avg"]["recall"]:.2f}', f'{report["macro avg"]["f1-score"]:.2f}', ''])
data.append(['weighted avg', f'{report["weighted avg"]["precision"]:.2f}',
             f'{report["weighted avg"]["recall"]:.2f}', f'{report["weighted avg"]["f1-score"]:.2f}', ''])

table = ax.table(cellText=data,
                 colLabels=['Clase', 'Precision', 'Recall', 'F1-Score', 'Support'],
                 cellLoc='center',
                 loc='center',
                 colWidths=[0.2, 0.2, 0.2, 0.2, 0.2])

table.auto_set_font_size(False)
table.set_fontsize(11)
table.scale(1.2, 1.8)

for (row, col), cell in table.get_celld().items():
    if row == 0:
        cell.set_facecolor('#2196F3')
        cell.set_text_props(color='white', fontweight='bold')
    elif row >= 6:
        cell.set_facecolor('#FFC107')
        cell.set_text_props(fontweight='bold')
    else:
        cell.set_facecolor('#f0f0f0' if row % 2 == 0 else 'white')

ax.set_title('Reporte de Clasificacion Detallado', fontsize=14, fontweight='bold', pad=20)
plt.tight_layout()
plt.savefig('img/10_reporte_clasificacion.png', dpi=150, bbox_inches='tight')
plt.close()
print("   Guardada: img/10_reporte_clasificacion.png")

# ============================================
# 3. MATRIZ DE CONFUSION
# ============================================
print("3. Generando matriz de confusion...")
plt.figure(figsize=(10, 8))
cm = confusion_matrix(y_true, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=labels, yticklabels=labels,
            annot_kws={"size": 14})
plt.title('Matriz de Confusion: Realidad vs Prediccion', fontsize=14, fontweight='bold')
plt.xlabel('Prediccion de la IA', fontsize=12)
plt.ylabel('Realidad (Etiqueta)', fontsize=12)
plt.tight_layout()
plt.savefig('img/11_matriz_confusion.png', dpi=150, bbox_inches='tight')
plt.close()
print("   Guardada: img/11_matriz_confusion.png")

# ============================================
# 4. VISUALIZACION DE PREDICCIONES
# ============================================
print("4. Generando visualizacion de predicciones...")
plt.figure(figsize=(18, 12))
plt.suptitle('Predicciones del Modelo (Verde=Correcto, Rojo=Error)', fontsize=16, fontweight='bold')

for images, labels_batch in val_ds.take(1):
    preds = model_ready.predict(images, verbose=0)
    
    for i in range(15):
        actual = labels[labels_batch[i]]
        pred_idx = np.argmax(preds[i])
        predicho = labels[pred_idx]
        confianza = 100 * np.max(preds[i])
        
        color = 'green' if actual == predicho else 'red'
        
        plt.subplot(3, 5, i + 1)
        plt.imshow(images[i].numpy().astype("uint8"))
        plt.title(f"REAL: {actual}\nPRED: {predicho}\n({confianza:.1f}%)", 
                 color=color, fontsize=10, fontweight='bold')
        plt.axis("off")

plt.tight_layout()
plt.savefig('img/12_predicciones.png', dpi=150, bbox_inches='tight')
plt.close()
print("   Guardada: img/12_predicciones.png")

# ============================================
# 5. PRUEBA CON DATASET TEST
# ============================================
print("5. Generando prueba con dataset test...")
test_files = sorted([f for f in os.listdir(PATH_TEST) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])

fig, axes = plt.subplots(4, 5, figsize=(20, 16))
fig.suptitle('Prueba con Dataset Test (20 imagenes externas)', fontsize=16, fontweight='bold', y=0.98)

resultados_test = {'correcto': 0, 'error': 0}

for i, filename in enumerate(test_files[:20]):
    row = i // 5
    col = i % 5
    
    img_path = os.path.join(PATH_TEST, filename)
    img = tf.keras.utils.load_img(img_path, target_size=IMG_SIZE)
    img_array = tf.keras.utils.img_to_array(img)
    img_batch = np.expand_dims(img_array, axis=0)
    
    preds = model_ready.predict(img_batch, verbose=0)
    pred_idx = np.argmax(preds[0])
    clase_predicha = labels[pred_idx]
    confianza = 100 * preds[0][pred_idx]
    
    # Determinar clase real del nombre de archivo
    fname_norm = filename.lower().replace("-", "").replace("_", "")
    clase_real = "Desconocido"
    for nombre_clase in labels:
        clase_norm = nombre_clase.lower().replace("-", "").replace("_", "")
        if clase_norm in fname_norm:
            clase_real = nombre_clase
            break
    
    es_correcto = (clase_real == clase_predicha)
    if es_correcto:
        resultados_test['correcto'] += 1
    else:
        resultados_test['error'] += 1
    
    color = 'green' if es_correcto else 'red'
    
    axes[row, col].imshow(img)
    axes[row, col].set_title(f'Real: {clase_real}\nPred: {clase_predicha}\n({confianza:.1f}%)', 
                             color=color, fontsize=10, fontweight='bold')
    axes[row, col].axis('off')

plt.tight_layout()
plt.savefig('img/13_prueba_test.png', dpi=150, bbox_inches='tight')
plt.close()
print("   Guardada: img/13_prueba_test.png")

# ============================================
# 6. RESUMEN DE RESULTADOS
# ============================================
print("6. Generando resumen de resultados...")
fig, ax = plt.subplots(figsize=(10, 6))
ax.axis('off')

data = [
    ['Accuracy Final', f'{report["accuracy"]*100:.1f}%'],
    ['Mejor Clase (F1)', f'{max(labels, key=lambda x: report[x]["f1-score"])} ({max(report[c]["f1-score"] for c in labels):.2f})'],
    ['Peor Clase (F1)', f'{min(labels, key=lambda x: report[x]["f1-score"])} ({min(report[c]["f1-score"] for c in labels):.2f})'],
    ['Test Correctos', f'{resultados_test["correcto"]}/20 ({resultados_test["correcto"]*5}%)'],
    ['Test Errores', f'{resultados_test["error"]}/20 ({resultados_test["error"]*5}%)'],
    ['Total Imagenes', '3,555'],
    ['Epocas Totales', '20 (10 + 10)'],
]

table = ax.table(cellText=data,
                 colLabels=['Metrica', 'Valor'],
                 cellLoc='center',
                 loc='center',
                 colWidths=[0.4, 0.4])

table.auto_set_font_size(False)
table.set_fontsize(12)
table.scale(1.2, 1.8)

for (row, col), cell in table.get_celld().items():
    if row == 0:
        cell.set_facecolor('#4CAF50')
        cell.set_text_props(color='white', fontweight='bold')
    else:
        cell.set_facecolor('#f0f0f0' if row % 2 == 0 else 'white')

ax.set_title('Resumen de Resultados del Modelo', fontsize=14, fontweight='bold', pad=20)
plt.tight_layout()
plt.savefig('img/14_resumen_resultados.png', dpi=150, bbox_inches='tight')
plt.close()
print("   Guardada: img/14_resumen_resultados.png")

print("\n" + "="*60)
print("TODAS LAS GRAFICAS GENERADAS EXITOSAMENTE!")
print("="*60)
print(f"\nArchivos en img/:")
for f in sorted(os.listdir('img')):
    if f.endswith('.png'):
        size = os.path.getsize(f'img/{f}') / 1024
        print(f"  - {f} ({size:.1f} KB)")
