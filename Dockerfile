# Uporabimo uradno PyTorch sliko z NVIDIA CUDA podporo
FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime

# Nastavimo delovno mapo v kontejnerju
WORKDIR /workspace

# Namestimo sistemske knjižnice (npr. git, če bi ga rabili)
RUN apt-get update && apt-get install -y git && rm -rf /var/lib/apt/lists/*

# Kopiramo requirements.txt in namestimo Python knjižnice
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Kopiramo vso ostalo kodo v kontejner
COPY . .

# Nastavimo, da se skripte lahko izvajajo
RUN chmod +x run_train.py run_test.py run_inference.py