import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from PIL import Image
import seaborn as sns

# Configuracion
PATH_BASE = "./dataset-cazas"
PATH_TEST = "./dataset-test"

# ============================================
# 1. DISTRIBUCION DEL DATASET (BARRAS)
# ============================================
print("1. Generando distribucion del dataset...")
folders = sorted([f for f in os.listdir(PATH_BASE) if os.path.isdir(os.path.join(PATH_BASE, f))])
counts = []
for name in folders:
    folder_path = os.path.join(PATH_BASE, name)
    files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    counts.append(len(files))

fig, ax = plt.subplots(figsize=(10, 6))
colors = ['#2196F3', '#4CAF50', '#FF9800', '#E91E63', '#9C27B0']
bars = ax.bar(folders, counts, color=colors, edgecolor='black', linewidth=0.5)

for bar, count in zip(bars, counts):
    ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 5,
            f'{count}', ha='center', va='bottom', fontweight='bold', fontsize=12)

ax.set_xlabel('Clase del Caza', fontsize=12, fontweight='bold')
ax.set_ylabel('Numero de Imagenes', fontsize=12, fontweight='bold')
ax.set_title('Distribucion del Dataset por Clase', fontsize=14, fontweight='bold')
ax.set_ylim(0, max(counts) + 50)
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig('img/02_distribucion_dataset.png', dpi=150, bbox_inches='tight')
plt.close()
print("   Guardada: img/02_distribucion_dataset.png")

# ============================================
# 2. EJEMPLO DE IMAGENES POR CLASE (4 POR CLASE)
# ============================================
print("2. Generando ejemplo de imagenes por clase...")
fig, axes = plt.subplots(5, 4, figsize=(16, 20))
fig.suptitle('Ejemplo de Imagenes por Clase (4 por clase)', fontsize=16, fontweight='bold', y=0.98)

for i, name in enumerate(folders):
    folder_path = os.path.join(PATH_BASE, name)
    files = sorted([f for f in os.listdir(folder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
    
    for j in range(4):
        if j < len(files):
            img_path = os.path.join(folder_path, files[j])
            img = mpimg.imread(img_path)
            axes[i, j].imshow(img)
            axes[i, j].set_title(f'{name}\n{files[j][:15]}...', fontsize=9)
        else:
            axes[i, j].text(0.5, 0.5, 'Sin imagen', ha='center', va='center')
        axes[i, j].axis('off')

plt.tight_layout()
plt.savefig('img/03_ejemplos_por_clase.png', dpi=150, bbox_inches='tight')
plt.close()
print("   Guardada: img/03_ejemplos_por_clase.png")

# ============================================
# 3. PRUEBA CON DATASET TEST
# ============================================
print("3. Generando prueba con dataset test...")
test_files = sorted([f for f in os.listdir(PATH_TEST) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])

fig, axes = plt.subplots(4, 5, figsize=(18, 16))
fig.suptitle('Prueba con Dataset Test (20 imagenes)', fontsize=16, fontweight='bold', y=0.98)

for i, filename in enumerate(test_files[:20]):
    row = i // 5
    col = i % 5
    
    img_path = os.path.join(PATH_TEST, filename)
    img = mpimg.imread(img_path)
    
    axes[row, col].imshow(img)
    # Extraer nombre de clase del archivo
    clase = filename.split('_')[0].upper()
    axes[row, col].set_title(f'{clase}\n{filename}', fontsize=10, color='blue')
    axes[row, col].axis('off')

plt.tight_layout()
plt.savefig('img/06_prueba_dataset_test.png', dpi=150, bbox_inches='tight')
plt.close()
print("   Guardada: img/06_prueba_dataset_test.png")

# ============================================
# 4. ANALISIS DE TAMANOS DE IMAGENES
# ============================================
print("4. Generando analisis de tamanos...")
widths = []
heights = []
ratios = []

for name in folders:
    folder_path = os.path.join(PATH_BASE, name)
    files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    for f in files[:50]:  # Muestra de 50 por clase
        try:
            img_path = os.path.join(folder_path, f)
            with Image.open(img_path) as img:
                w, h = img.size
                widths.append(w)
                heights.append(h)
                ratios.append(w/h)
        except:
            pass

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle('Analisis de Dimensiones de Imagenes', fontsize=14, fontweight='bold')

axes[0].hist(widths, bins=30, color='#2196F3', edgecolor='black', alpha=0.7)
axes[0].set_xlabel('Ancho (px)')
axes[0].set_ylabel('Frecuencia')
axes[0].set_title('Distribucion de Anchos')
axes[0].axvline(np.mean(widths), color='red', linestyle='--', label=f'Media: {np.mean(widths):.0f}px')
axes[0].legend()

axes[1].hist(heights, bins=30, color='#4CAF50', edgecolor='black', alpha=0.7)
axes[1].set_xlabel('Alto (px)')
axes[1].set_ylabel('Frecuencia')
axes[1].set_title('Distribucion de Altos')
axes[1].axvline(np.mean(heights), color='red', linestyle='--', label=f'Media: {np.mean(heights):.0f}px')
axes[1].legend()

axes[2].hist(ratios, bins=30, color='#FF9800', edgecolor='black', alpha=0.7)
axes[2].set_xlabel('Relacion Ancho/Alto')
axes[2].set_ylabel('Frecuencia')
axes[2].set_title('Distribucion de Relacion de Aspecto')
axes[2].axvline(np.mean(ratios), color='red', linestyle='--', label=f'Media: {np.mean(ratios):.2f}')
axes[2].legend()

plt.tight_layout()
plt.savefig('img/07 analisis_tamanos.png', dpi=150, bbox_inches='tight')
plt.close()
print("   Guardada: img/07 analisis_tamanos.png")

# ============================================
# 5. TABLA RESUMEN
# ============================================
print("5. Generando tabla resumen...")

fig, ax = plt.subplots(figsize=(10, 4))
ax.axis('off')

data = []
for i, (name, count) in enumerate(zip(folders, counts)):
    data.append([name, count, f'{count/sum(counts)*100:.1f}%'])

table = ax.table(cellText=data, 
                 colLabels=['Clase', 'Imagenes', 'Porcentaje'],
                 cellLoc='center',
                 loc='center',
                 colWidths=[0.3, 0.3, 0.3])

table.auto_set_font_size(False)
table.set_fontsize(12)
table.scale(1.2, 1.5)

for (row, col), cell in table.get_celld().items():
    if row == 0:
        cell.set_facecolor('#2196F3')
        cell.set_text_props(color='white', fontweight='bold')
    else:
        cell.set_facecolor('#f0f0f0' if row % 2 == 0 else 'white')

ax.set_title('Resumen del Dataset', fontsize=14, fontweight='bold', pad=20)
plt.tight_layout()
plt.savefig('img/08_tabla_resumen.png', dpi=150, bbox_inches='tight')
plt.close()
print("   Guardada: img/08_tabla_resumen.png")

print("\n" + "="*50)
print("TODAS LAS GRAFICAS GENERADAS!")
print("="*50)
print("\nArchivos en img/:")
for f in sorted(os.listdir('img')):
    if f.endswith('.png'):
        print(f"  - {f}")
