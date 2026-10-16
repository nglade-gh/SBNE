import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import glob
import re
import os
import itertools

# ==========================================
# 1. GÉNÉRATEURS DE GRAPHIQUES
# ==========================================

def generer_dynamique_temporelle(df_total, dossier_sauvegarde):
    """Génère et sauvegarde le graphique d'évolution sur toute la simulation."""
    colonnes_phenotype = [
        "Oeil_Avancer", "Oeil_Rotation",
        "Avancer_Rotation", "Rotation_Avancer",
        "Avancer_Oeil", "Rotation_Oeil"
    ]
    
    df_melted = df_total.melt(
        id_vars=['Iteration'], 
        value_vars=colonnes_phenotype,
        var_name='Connexion', 
        value_name='Poids'
    )

    # Le graphique global (celui qui affiche tout).
    plt.figure(figsize=(16, 8))
    sns.lineplot(data=df_melted, x='Iteration', y='Poids', hue='Connexion', linewidth=2, errorbar=None)

    plt.title("Dynamique Évolutive Globale des Réseaux Neuronaux", fontsize=16, fontweight='bold')
    plt.xlabel("Itérations", fontsize=12)
    plt.ylabel("Poids d'influence moyen", fontsize=12)
    plt.axhline(0, color='black', linestyle='--', alpha=0.5, linewidth=1.5)
    plt.grid(True, alpha=0.3)
    plt.legend(bbox_to_anchor=(1.01, 1), loc='upper left')
    plt.tight_layout()
    
    chemin_global = os.path.join(dossier_sauvegarde, "evolution_globale.png")
    plt.savefig(chemin_global, dpi=150)
    plt.close()

    # Les graphiques de comparaison (Deux à deux)
    paires = list(itertools.combinations(colonnes_phenotype, 2))
    total_paires = len(paires)
    print(f"  Génération de {len(paires)} graphiques de comparaison...")
    for i, (trait_A, trait_B) in enumerate(paires, start=1):
        # On filtre les données pour ne garder que les deux traits ciblés
        df_paire = df_melted[df_melted['Connexion'].isin([trait_A, trait_B])]
        
        plt.figure(figsize=(12, 6))
        sns.lineplot(data=df_paire, x='Iteration', y='Poids', hue='Connexion', linewidth=2, palette=['#1f77b4', '#ff7f0e'], errorbar=None)
        
        plt.title(f"Compétition Évolutive : {trait_A} vs {trait_B}", fontsize=14, fontweight='bold')
        plt.xlabel("Itérations", fontsize=12)
        plt.ylabel("Poids d'influence moyen", fontsize=12)
        plt.axhline(0, color='black', linestyle='--', alpha=0.5, linewidth=1.5)
        plt.grid(True, alpha=0.3)
        plt.legend(bbox_to_anchor=(1.01, 1), loc='upper left')
        plt.tight_layout()
        
        # Sauvegarde
        nom_fichier = f"comparaison_{trait_A}_vs_{trait_B}.png"
        chemin_paire = os.path.join(dossier_sauvegarde, nom_fichier)
        plt.savefig(chemin_paire, dpi=150)
        plt.close()
        
        # Affichage de la progression
        print(f"[{i}/{total_paires}] Comparaison {trait_A} vs {trait_B} terminée.")


def generer_cartes_spatiales(df, iteration, chemin_sauvegarde, width=1280, height=720):
    """Génère et sauvegarde la mosaïque spatiale pour une époque précise."""
    traits_phenotypiques = [
        "Oeil_Avancer", "Oeil_Rotation",
        "Avancer_Rotation", "Rotation_Avancer",
        "Avancer_Oeil", "Rotation_Oeil"
    ]

    fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(20, 12))
    fig.suptitle(f"Cartes Écologiques (Itération {iteration})\n{len(df)} agents", fontsize=18, fontweight='bold')
    axes = axes.flatten()
    
    for i, trait in enumerate(traits_phenotypiques):
        ax = axes[i]
        if trait not in df.columns:
            continue
            
        scatter = ax.scatter(
            x=df['Pos_X'], y=df['Pos_Y'], c=df[trait], 
            cmap='coolwarm', s=40, alpha=0.8, edgecolors='black', linewidth=0.5
        )
        
        ax.set_title(f"Trait : {trait}", fontsize=14, fontweight='bold')
        ax.set_xlim(0, width)
        ax.set_ylim(0, height)
        ax.invert_yaxis()
        ax.grid(True, linestyle='--', alpha=0.3)
        fig.colorbar(scatter, ax=ax, fraction=0.046, pad=0.04)

    plt.tight_layout(rect=(0, 0, 1, 0.95)) #[] en () le tuple plait mieux à matplotlib
    plt.savefig(chemin_sauvegarde, dpi=150)
    plt.close()


def generer_barplot_moyennes(df, iteration, chemin_sauvegarde):
    """Génère et sauvegarde le barplot des moyennes pour une époque précise."""
    colonnes_phenotype = [
        "Oeil_Avancer", "Oeil_Rotation",
        "Avancer_Rotation", "Rotation_Avancer",
        "Avancer_Oeil", "Rotation_Oeil"
    ]
    
    df_melted = df.melt(value_vars=colonnes_phenotype, var_name='Connexion', value_name='Poids')

    plt.figure(figsize=(10, 6))
    sns.barplot(data=df_melted, x='Connexion', y='Poids', hue='Connexion', palette='magma', legend=False, capsize=0.1)

    plt.title(f"Moyenne des influences synaptiques (Itération {iteration})", fontsize=15, fontweight='bold')
    plt.xlabel("Paire de connexion", fontsize=12)
    plt.ylabel("Poids d'influence moyen", fontsize=12)
    plt.axhline(0, color='red', linestyle='--', alpha=0.5, linewidth=2) 
    plt.xticks(rotation=15, fontsize=11)
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    
    plt.savefig(chemin_sauvegarde, dpi=150)
    plt.close()

# ==========================================
# 2. PIPELINE
# ==========================================

def lancer_pipeline_analyse(dossier_simulation):
    print("="*50)
    print(f"LANCEMENT DU PIPELINE D'ANALYSE")
    print(f"Cible : {dossier_simulation}")
    print("="*50)

    # Définition et création de l'arborescence
    dossier_exports = os.path.join(dossier_simulation, "exports")
    dossier_analyse = os.path.join(dossier_simulation, "analyse")
    
    dir_temporel = os.path.join(dossier_analyse, "1_dynamique_temporelle")
    dir_spatial = os.path.join(dossier_analyse, "2_cartes_spatiales")
    dir_moyennes = os.path.join(dossier_analyse, "3_moyennes_barplot")
    
    for dossier in [dir_temporel, dir_spatial, dir_moyennes]:
        os.makedirs(dossier, exist_ok=True)

    # Récupération des fichiers
    fichiers = glob.glob(os.path.join(dossier_exports, "agents_save_*.csv"))
    if not fichiers:
        print(f"❌ Aucun fichier CSV trouvé dans {dossier_exports}")
        return
        
    print(f"✅ {len(fichiers)} fichiers détectés. Début du traitement...")

    liste_df_globale = []

    # Boucle de traitement fichier par fichier
    for index, fichier in enumerate(fichiers):
        match = re.search(r'agents_save_(\d+)\.csv', fichier)
        if match:
            iteration = int(match.group(1))
            
            # Chargement du DataFrame
            df_courant = pd.read_csv(fichier)
            df_courant['Iteration'] = iteration
            liste_df_globale.append(df_courant)
            
            # Chemins de sauvegarde
            chemin_spatial = os.path.join(dir_spatial, f"spatial_{iteration:08d}.png")
            chemin_moyenne = os.path.join(dir_moyennes, f"moyennes_{iteration:08d}.png")
            
            # Génération des graphiques individuels
            generer_cartes_spatiales(df_courant, iteration, chemin_spatial)
            generer_barplot_moyennes(df_courant, iteration, chemin_moyenne)
            
            print(f"  [{index+1}/{len(fichiers)}] Itération {iteration} traitée.")

    # 4. Traitement temporel global
    print("\nFusion des données pour l'analyse temporelle...")
    df_total = pd.concat(liste_df_globale, ignore_index=True)
    df_total.sort_values(by='Iteration', inplace=True)
    
    chemin_temporel = os.path.join(dir_temporel, "evolution_globale.png")
    generer_dynamique_temporelle(df_total, dir_temporel)
    print("  Évolution globale générée.")

    print("\n" + "="*50)
    print("ANALYSE TERMINÉE AVEC SUCCÈS")
    print(f"Tous les graphiques sont rangés dans : {dossier_analyse}")
    print("="*50)

# if __name__ == "__main__":
#     sns.set_theme(style="whitegrid")
#     DOSSIER_SIMULATION = "results/No_weights_cap_higher_cost_neuron_and_10_foods"
#     lancer_pipeline_analyse(DOSSIER_SIMULATION)

if __name__ == "__main__":
    sns.set_theme(style="whitegrid")

    # Chemin racine des simulations
    DOSSIER_RACINE = "simulations"

    # Parcourir tous les sous-dossiers de simulations/
    for dossier_simulation in os.listdir(DOSSIER_RACINE):
        chemin_simulation = os.path.join(DOSSIER_RACINE, dossier_simulation)

        # Vérifier si c'est un dossier et s'il contient un sous-dossier "exports"
        if os.path.isdir(chemin_simulation) and os.path.exists(os.path.join(chemin_simulation, "exports")):
            print(f"\nAnalyse du dossier : {dossier_simulation}")
            lancer_pipeline_analyse(chemin_simulation)