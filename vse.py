import os, time

def zazeni(cmd):
    print(f"\n Zaganjam: {cmd}")
    os.system(f"python3 {cmd}")

# 1. Priprava ni potrebna (beremo direktno)
zazeni("Dataset.py")

# 2. Trening
zazeni("run_train.py")

# 3. Inferenca
zazeni("run_inference.py")

# 4. Test
zazeni("run_test.py")

zazeni("Vizualizacija.py")

print("\n Končano")