import argparse
import os
import sys
import torch

# 1. Dodamo pot do mape mednext_lib, da Python najde tvoje datoteke
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(current_dir, 'mednext_lib'))

# 2. Poskusimo uvoziti model
try:
    # Pazi: Ime datoteke je MedNextV1 (kot na tvoji sliki), razred notri pa je verjetno MedNeXt
    from MedNextV1 import MedNeXt
    print("✅ USPEH: MedNeXt model je bil uspešno najden in uvožen!")
except ImportError as e:
    print(f"❌ NAPAKA pri uvozu: {e}")
    print("Preveri, če je datoteka 'MedNextV1.py' res v mapi 'mednext_lib'.")
    sys.exit(1)

def parse_args():
    parser = argparse.ArgumentParser(description='Trening skripta za MedNeXt')
    # Argumenti, ki jih zahteva profesor
    parser.add_argument('--data_path', type=str, required=True, help='Pot do podatkov')
    parser.add_argument('--params_path', type=str, required=True, help='Pot do params.json')
    parser.add_argument('--output_path', type=str, required=True, help='Pot za shranjevanje modela')
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Preverimo GPU
    if torch.cuda.is_available():
        print(f"✅ GPU zaznan: {torch.cuda.get_device_name(0)}")
    else:
        print("⚠️ Opozorilo: GPU ni zaznan, trening bo zelo počasen (CPU).")

    print("-" * 30)
    print("Pripravljen na trening z nastavitvami:")
    print(f"  📂 Podatki:   {args.data_path}")
    print(f"  ⚙️  Parametri: {args.params_path}")
    print(f"  💾 Izhod:     {args.output_path}")
    print("-" * 30)

    # Tu bomo kasneje ustvarili model:
    # model = MedNeXt(...)

if __name__ == '__main__':
    main()