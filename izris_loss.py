import matplotlib.pyplot as plt

# --- TVOJI PODATKI (Prepisani iz tvojega loga) ---
losses = [
    0.2149, 0.1503, 0.1425, 0.1355, 0.1319, 0.1279, 0.1249, 0.1219, 0.1198, 0.1169,
    0.1147, 0.1123, 0.1094, 0.1067, 0.1054, 0.1027, 0.1017, 0.0984, 0.0960, 0.0944,
    0.0928, 0.0918, 0.0900, 0.0894, 0.0861, 0.0852, 0.0843, 0.0827, 0.0822, 0.0816,
    0.0794, 0.0785, 0.0774, 0.0766, 0.0745, 0.0741, 0.0741, 0.0722, 0.0721, 0.0720,
    0.0696, 0.0674, 0.0674, 0.0690, 0.0668, 0.0647, 0.0650, 0.0648, 0.0642, 0.0636,
    0.0623, 0.0613, 0.0604, 0.0600, 0.0590, 0.0598, 0.0578, 0.0569, 0.0569, 0.0565,
    0.0554, 0.0559, 0.0560, 0.0539, 0.0539, 0.0543, 0.0530, 0.0517, 0.0523, 0.0527,
    0.0519, 0.0500, 0.0500, 0.0501, 0.0495, 0.0497, 0.0489, 0.0482, 0.0493, 0.0473,
    0.0467, 0.0462, 0.0466, 0.0464, 0.0459, 0.0455, 0.0451, 0.0445, 0.0443
]

# Ustvarimo seznam epoh (od 1 do 89)
epochs = range(1, len(losses) + 1)

# --- IZRIS GRAFA ---
plt.figure(figsize=(10, 6))

# Narišemo črto
plt.plot(epochs, losses, label='Training Loss', color='blue', linewidth=2)

# Dodamo naslove
plt.title('Potek učenja modela (Training Loss)', fontsize=16)
plt.xlabel('Epohe', fontsize=12)
plt.ylabel('Izguba (Loss)', fontsize=12)

# Dodamo mrežo za lažje branje
plt.grid(True, linestyle='--', alpha=0.7)

# Oznacimo začetno in končno točko
plt.scatter(1, losses[0], color='red', zorder=5)
plt.text(1, losses[0], f'{losses[0]:.4f}', verticalalignment='bottom')

plt.scatter(len(losses), losses[-1], color='green', zorder=5)
plt.text(len(losses), losses[-1], f'{losses[-1]:.4f}', verticalalignment='top')

plt.legend()
plt.tight_layout()

# Shranimo sliko
plt.savefig("Graf_Loss.png", dpi=300)
print("✅ Graf je shranjen kot 'Graf_Loss.png'")