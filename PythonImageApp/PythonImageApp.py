# -*- coding: utf-8 -*-
import sys  # Moduł sys - do obsługi błędów i wyjścia
import cv2  #OpenCV - przetwarzanie obrazów
import numpy as np  #NumPy - operacje na tablicach / bajtach
import urllib.request   #urllib.request - pobieranie danych z URL
import matplotlib.pyplot as plt #Matplotlib - wyświetlanie obrazów

# Wczytanie obrazu
url = "https://upload.wikimedia.org/wikipedia/commons/3/35/AC_Cobra_GT_Roadster_04.jpg"

# Dodanie nagłówka User-Agent, żeby serwer pozwolił pobrać obraz
req = urllib.request.Request(
    url, 
    headers={'User-Agent': 'Mozilla/5.0'}
)

# Pobranie obrazu z URL (z obsługą wyjątków)
try:
    with urllib.request.urlopen(req, timeout=10) as resp:
        image_data = np.asarray(bytearray(resp.read()), dtype=np.uint8)
except Exception as e:
    print("Błąd pobierania obrazu:", e, file=sys.stderr)
    sys.exit(1)

# Dekodowanie obrazu (sprawdzenie poprawności)
image = cv2.imdecode(image_data, cv2.IMREAD_COLOR)  # BGR
if image is None:
    print("Nie udało się zdekodować obrazu.", file=sys.stderr)
    sys.exit(1)

# Wyświetlenie oryginalnego obrazu
# Jeśli obraz jest kolorowy: konwersja BGR -> RGB (Matplotlib używa RGB)
if image.ndim == 3 and image.shape[2] == 3:
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    plt.figure(figsize=(6,6))
    plt.title("Oryginalny obraz")
    plt.imshow(image_rgb)
else:
    plt.figure(figsize=(6,6))
    plt.title("Oryginalny obraz (szary)")
    plt.imshow(image, cmap='gray')
plt.axis("off")
plt.show()

# Zmiana rozdzielczości (zmniejszenie o 50%)
height, width = image.shape[:2]
new_width = max(1, width // 2)
new_height = max(1, height // 2)
# Użycie INTER_AREA do zmniejszania (lepsze jakościowo)
resized = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)

# Konwersja do skali szarości
if resized.ndim == 3 and resized.shape[2] == 3:
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
else:
    gray = resized

# Obrót obrazu o 90 stopni
rotated = cv2.rotate(gray, cv2.ROTATE_90_CLOCKWISE)

# Wyświetlenie obrazu wynikowego
plt.figure(figsize=(5,5))
plt.title("Obraz po przetworzeniu (szary + 90°)")
plt.imshow(rotated, cmap='gray')
plt.axis("off")
plt.show()

# Wyświetlenie macierzy obrazu
print("Macierz obrazu (fragment):")
print(rotated[:10, :10])  # fragment 10x10 pikseli
print("\nWymiary obrazu po przetworzeniu:", rotated.shape)