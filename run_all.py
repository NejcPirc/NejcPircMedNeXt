import os
import time

print("=========================================")
print("🚀 ZAGANJAM CELOTEN PIPELINE MEDNEXT")
print("=========================================")

print("\n[1/3] TRENING ...")
os.system("python3 run_train.py")

print("\n[2/3] INFERENCA (NAPOVEDOVANJE) ...")
os.system("python3 run_inference.py")

print("\n[3/3] TESTIRANJE (OCENJEVANJE) ...")
os.system("python3 run_test.py")

print("\n=========================================")
print("✅ VSE KONČANO!")
print("=========================================")