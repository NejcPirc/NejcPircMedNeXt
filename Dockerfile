# Uporabimo uradno PyTorch sliko
FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime

# Nastavimo delovno mapo
WORKDIR /workspace

# Namestimo sistemske knjižnice
# Odstrani libpng, libjpeg itd. Pusti samo git.
RUN apt-get update && apt-get install -y git && rm -rf /var/lib/apt/lists/*

ENV PYTHONUNBUFFERED=1

# Kopiramo requirements in namestimo Python pakete
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Kopiramo celotno kodo projekta v kontejner
COPY . .

# Povemo, da naj se skripte lahko izvajajo
RUN chmod +x *.py

# Privzeto zaženemo celoten pipeline (lahko pa uporabnik povozi ta ukaz)
CMD ["python3", "-u", "vse.py"]