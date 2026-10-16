from evolution_tracker import rebuild_network

# Connexion à la base de données
db_path = "evolution.db"

# Exemple 1 : Reconstruire le cerveau d'un agent à une itération donnée
agent_id = 10  # Remplace par l'ID d'un agent qui t'intéresse
target_iteration = 60000  # Remplace par l'itération qui t'intéresse

network = rebuild_network(db_path, agent_id, target_iteration)
print(f"🧠 Cerveau de l'agent {agent_id} à l'itération {target_iteration} :")
print("States:", network["states"])
print("Weights:", network["weights"][:2])  # Affiche les 2 premières lignes des poids pour éviter la surcharge

# Exemple 2 : Comparer le cerveau d'un agent à deux itérations différentes
iteration_1 = 0
iteration_2 = 50000
network_1 = rebuild_network(db_path, agent_id, iteration_1)
network_2 = rebuild_network(db_path, agent_id, iteration_2)

print(f"\n🔄 Évolution de l'agent {agent_id} entre {iteration_1} et {iteration_2} :")
print("States à l'itération 0:", network_1["states"])
print("States à l'itération 50000:", network_2["states"])