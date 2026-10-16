import os
import json
import sqlite3
from datetime import datetime

def setup_simulation(test_name=None, base_dir="simulations"):
    """
    Crée un dossier pour une nouvelle simulation avec :
    - Un sous-dossier unique (ex: sim_20260709_120000/)
    - Un fichier evolution.db initialisé
    - Un fichier config.json avec les paramètres

    Args:
        test_name (str): Nom personnalisé pour la simulation (optionnel).
                        Si None, utilise un timestamp.
        base_dir (str): Dossier de base pour les simulations (par défaut: "simulations").

    Returns:
        dict: Chemin vers le dossier, la base de données, et les paramètres.
    """
    # 1. Génère un nom unique pour la simulation
    if test_name is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        sim_name = f"sim_{timestamp}"
    else:
        sim_name = test_name

    # 2. Crée le dossier de la simulation
    sim_dir = os.path.join(base_dir, sim_name)
    os.makedirs(sim_dir, exist_ok=True)

    # 3. Crée le chemin vers la base de données
    db_path = os.path.join(sim_dir, "evolution.db")

    # 4. Initialise la base de données
    conn = sqlite3.connect(db_path)
    with open("init_evolution_db.sql", "r") as f:
        conn.executescript(f.read())
    conn.close()

    # 5. Crée un fichier config.json avec les paramètres
    config = {
        "TEST_NAME": sim_name,
        "DB_PATH": db_path,
        "CREATED_AT": datetime.now().isoformat(),
        "SIMULATION_DIR": sim_dir,
    }

    config_path = os.path.join(sim_dir, "config.json")
    with open(config_path, "w") as f:
        json.dump(config, f, indent=4)

    # 6. Retourne les chemins utiles
    return {
        "sim_dir": sim_dir,
        "db_path": db_path,
        "config_path": config_path,
        "test_name": sim_name,
    }

# Exemple d'utilisation
if __name__ == "__main__":
    # Initialise une nouvelle simulation
    simulation = setup_simulation(test_name=None)  # ou test_name="ma_simulation_test"

    print(f"📁 Dossier de simulation : {simulation['sim_dir']}")
    print(f"💾 Base de données : {simulation['db_path']}")
    print(f"📄 Fichier de config : {simulation['config_path']}")