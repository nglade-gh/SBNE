# SBNE — Simulateur d'écosystème évolutionnaire

Simulation d'un écosystème de **micro-organismes agentifs** dont le « cerveau » est un
**réseau de neurones booléen signé** (SBN, *signed boolean network*). Les agents se
déplacent, voient, mangent et se reproduisent dans un **environnement torique** (le
monde s'enroule : à droite du monde on ressort à gauche). Chaque cerveau est soumis à
mutation à la reproduction : c'est de là que vient l'évolution. L'historique complet des
cerveaux est enregistré en base SQLite **sous forme de deltas (comme git)**.

- Monde : `1280 × 720` px par défaut, torique, rendu pygame + bandeau d'information.
  La fenêtre fait `1600 × 770` : le monde + un **panneau latéral « cerveau »** (~1/4 de
  la largeur) qui affiche le SBN de l'agent suivi, directement dans la fenêtre (aucune
  fenêtre matplotlib bloquante).
- Population initiale : 150 agents (paramétrable), chacun avec ~1000 d'énergie.
- L'énergie **globale** du système (agents + nourriture) est **strictement conservée** à
  chaque pas : c'est l'invariant de contrôle du simulateur.

---

## 1. Comment fonctionne la simulation

### Les agents
Un agent possède : une position `(x, y)`, un angle, une **énergie** (qu'il dépense), un
**estomac** (énergie digérée en cours), et son **SBN**. Il meurt quand son énergie passe
sous **50** (`Agent.DEATH_ENERGY`) ; il se reproduit quand elle atteint
**1200** (`DIVISION_ENERGY`) : l'enfant est placé 15 px plus loin, dans la zone aveugle
du parent, avec l'énergie du parent et un cerveau **muté**.

### Le cerveau (SBN)
- Réseau de neurones **booléens signés** : états `0/1`, poids **entiers** bornés à
  `VALEUR_MAX_POIDS` (3).
- 4 **organes** = neurones fondamentaux, **non supprimables** (indices fixes) :

  | Index | Organe | Rôle |
  |---|---|---|
  | 0 | Œil (`EYE`) | Entrée : ce que l'agent voit dans son cône de vision |
  | 1 | Bouche (`MOUTH`) | Action : manger |
  | 2 | Nageoire F (`MOVE`) | Action : avancer |
  | 3 | Nageoire R (`ROTATE`) | Action : tourner |

- Le réseau démarre avec 8 nœuds ; les autres nœuds et arcs sont libres. À chaque
  naissance, le cerveau de l'enfant subit des mutations : **suppression** de nœud/arc
  (probabilité `PROBA_DELETION`), **insertion** (`PROBA_INSERTION`), **évolution de
  poids** (`PROBA_EVOLUTION`).
- À chaque pas : `état_sortie = (état @ poids > 0)` ; la sortie des organes donne les
  actions (manger, avancer, tourner).

### L'énergie et la nourriture
- Actions payantes : avancer (`COST_MOVE`), tourner (`COST_ROTATE`), manger (`COST_EAT`),
  chaque nœud coûte `COST_NEURON` par pas, plus le `COST_METABOLISM`.
- Manger : un agent peut dévorer une **nourriture** du sol ou un **autre agent** (duel
  : le plus énergique gagne ; à égalité, tirage aléatoire). Les conflits sont résolus en
  fin de pas (intentions de manger).
- La nourriture au sol est une liste de **particules** (`Food`) réparties spatialement
  par une **grille de diffusion** (`DiffusionGrid`) : quand un agent digeste
  (`stomach > DIGESTION_MIN`), les déchets sont déposés sur le centre + cases voisines,
  **inactifs** pendant `DIGESTION_INTERVAL` pas (le temps de se « décomposer »), puis
  mangeables. La diffusion ne crée ni ne détruit d'énergie (seulement la répartit).
- Un agent mort laisse son énergie sur le sol (même mécanisme).

### Un pas de simulation
1. Activation des déchets arrivés à échéance.
2. Mélange aléatoire des agents (équité : aucun ordre privilégié).
3. Reconstruction de la **grille spatiale** (recherche de voisins en O(1) au lieu d'O(n²),
   avec enroulement torique).
4. Pour chaque agent : **perception** (voisins dans le cône de vision), **pas du SBN**,
   actions (manger / avancer / tourner), **digestion**, éventuelle **division**
   (l'enfant agit au pas suivant) + enregistrement du cerveau muté en base.
5. Résolution des duels, nettoyage des morts (mark-and-sweep), dépôt des corps.

---

## 2. Structure du projet

| Fichier | Rôle |
|---|---|
| `Environnement.py` | **Cœur** : classe `Simulation` (boucle principale, orchestration logique/rendu/base). Point d'entrée. |
| `SbNetwork.py` | Le cerveau : réseau booléen signé, mutations, pas logique. |
| `Agent.py` | L'agent : état, perception, actions, énergie, division. |
| `Food.py` | Particule de nourriture (position, énergie, active/inactive). |
| `DiffusionGrid.py` | Grille de répartition spatiale des dépôts (nourriture initiale, déchets). |
| `SpatialGrid.py` | Index spatial torique (recherche de voisins). |
| `Interface.py` | Rendu pygame : bandeau d'information, pause, opacité des cônes. |
| `GraphVisualisation.py` | Graphiques matplotlib (résumé, graphe d'un SBN). |
| `evolution_tracker.py` | **Historique git-like** : snapshots + deltas jsonpatch en SQLite, reconstruction. |
| `ExportData.py` | Export des cerveaux (`save_nodes_influence`) toutes les 50 000 pas, dans `exports/`. |
| `launcher.py` | **Lanceur de campagnes parallèles** avec interface web (voir §7). |
| `Analyse.py` | Pipeline d'analyse des exports (graphiques pandas/seaborn/matplotlib). |
| `analyse_evolution.py` | Exemple : reconstruire le cerveau d'un agent à une itération. |
| `genealogy_viewer.py` / `genealogy_viewerv2.py` | Visualiseurs Tkinter de la généalogie (deux versions, la v2 est plus récente). |
| `get_agent_lineage_with_visualization.py` | Généalogie + visualisation networkx (utilisé par les viewers). |
| `lignéfixv2.py` | Variant de la généalogie (base `evolution.db` dans le dossier courant). |
| `save_network_state.py` | ⚠️ Ancien module de sauvegarde, **supplanté par `evolution_tracker.py`** (non importé par le simulateur). |
| `SpatialGridAgent.py` | ⚠️ Ancien index spatial, **doublon mort** de `SpatialGrid.py`. |
| `simulation_setup.py` | ⚠️ Script de setup historique, **cassé/inutilisé** (référence un SQL absent). |
| `*.copy.py` | **Backups des originaux** d'avant la réforme de 2026-10-16. Ne jamais écraser. |
| `simulations/` | Un sous-dossier par simulation (voir §5). |

---

## 3. Installation

### Installation complète (nouvelle machine)
```bash
cd SBNE
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

### Vous avez déjà un venv (ex. créé avec seulement 3 paquets)
Rien à recréer — il suffit de compléter les dépendances manquantes :
```bash
.venv/bin/pip install -r requirements.txt
```
C'est le correctif de l'erreur
`ModuleNotFoundError: No module named 'networkx'` : `requirements.txt` fait
autorité, et pip n'installe que ce qui manque.

### Les 7 dépendances (`requirements.txt`)

| Paquet | Utilisé pour |
|---|---|
| `numpy` | Les SBN (opérations matricielles), grille de diffusion, historique des réseaux |
| `pygame` | Fenêtre, boucle de simulation, rendu du monde torique |
| `jsonpatch` | L'historique git-like : calcul et rejeu des deltas de cerveaux |
| `seaborn` | Statistiques et graphiques de synthèse |
| `matplotlib` | Résumés, graphes de SBN, généalogies |
| `pandas` | L'analyse post-hoc (`Analyse.py`) |
| `networkx` | L'export des cerveaux (`ExportData.py`), graphes de généalogie |

Python ≥ 3.10 (testé sur 3.12 et 3.13, avec un **venv propre** sans
`--system-site-packages` : installation + run headless vérifiés).
**Toujours exécuter avec `.venv/bin/python`** (les dépendances ne sont pas dans le
Python système).

> ⚠️ **Le venv n'est pas déplaçable.** Les scripts de `.venv/bin/` ont des shebangs à
> chemin absolu : si vous déplacez ou recopiez le dossier du projet, le venv copie avec
> est cassé (`zsh: .venv/bin/pip: bad interpreter: ... no such file or directory`).
> Pire, `python3 -m venv .venv` sur un venv existant ne régénère pas les scripts.
> Solution : supprimer et recréer sur place —
> `rm -rf .venv && python3 -m venv .venv && .venv/bin/pip install -r requirements.txt`.

---

## 4. Lancer une simulation

### Avec fenêtre (défaut)
```bash
.venv/bin/python Environnement.py
```

### Sans fenêtre (calcul seul — pour les campagnes, CI, benchmarks)
```bash
.venv/bin/python Environnement.py --headless --max-steps 2000 --seed 7
```

### Options CLI

| Option | Effet |
|---|---|
| `--headless` | Pas de fenêtre (SDL en mode dummy), calcul seul. |
| `--max-steps N` | Arrêt automatique après N pas. **Prioritaire** sur `PARAMS["MAX_ITERATIONS"]`. |
| `--seed N` | Surcharge `PARAMS["SEED"]` (reproductibilité). |
| `--param CLE=VALEUR` | Surcharge un paramètre `PARAMS` (répétable). Ex. `--param NUM_AGENTS=300 --param PROBA_INSERTION=0.6`. Valeurs converties en int/float/bool/str ; clé inconnue → erreur (pas de faute de frappe silencieuse). |
| `--outdir NOM` | Nom du sous-dossier créé dans `simulations/` (défaut : horodatage). |

> **Temps maximal** : `PARAMS["MAX_ITERATIONS"]` (défaut `0` = illimité) est le
> paramètre de base ; `--max-steps` le surcharge s'il est passé.

### Raccourcis clavier (mode fenêtre)

| Touche | Action |
|---|---|
| `Espace` | Pause / lecture |
| `G` | Activer / désactiver le rendu (calcule plus vite) |
| `V` | Afficher / masquer les cônes de vision |
| `S` | Fenêtre de résumé des statistiques |
| Clic gauche sur un agent | **Suivi** de l'agent : son SBN s'affiche dans le panneau latéral (pause auto) |
| Clic gauche sur le vide | Ferme le panneau latéral (pause auto) |

---

## 5. Paramètres de la simulation (`PARAMS` dans `Environnement.py`)

| Clé | Défaut | Signification |
|---|---|---|
| `TEST_NAME` | — | Nom de l'expérience (affiché, tracé dans les exports). |
| `SEED` | `42` | Graine aléatoire (reproductibilité). |
| `NUM_AGENTS` | `150` | Population initiale. |
| `MAX_ITERATIONS` | `0` | **Temps maximal** : nombre max de pas (0 = illimité). |
| `BASE_ENERGY` | `1000` | Énergie initiale des agents. |
| `DIVISION_ENERGY` | `1200` | Seuil de reproduction. |
| `PROBA_DELETION` | `0.4` | Probabilité (à la naissance) de supprimer un nœud/arc. |
| `PROBA_INSERTION` | `0.4` | Probabilité d'insérer un nœud/arc. |
| `PROBA_EVOLUTION` | `0.4` | Probabilité d'évoluer un poids. |
| `VALEUR_MAX_POIDS` | `3` | Amplitude max des poids (entiers). |
| `DISTANCE_VISION` | `70` | Portée de la vision (px). |
| `VISION_ANGLE` | `30` | Demi-angle du cône de vision (degrés). |
| `COST_NEURON` | `0.3` | Coût par nœud, par pas. |
| `COST_MOVE` | `0.02` | Coût d'avancer. |
| `COST_ROTATE` | `0.1` | Coût de tourner. |
| `COST_EAT` | `0.01` | Coût de manger. |
| `COST_METABOLISM` | `0.1` | Coût de base par pas. |
| `NUM_FOOD` | `10` | Dépôts de nourriture initiaux (500 d'énergie chacun). |
| `DIGESTION_INTERVAL` | `120` | Pas avant activation des déchets au sol. |
| `DIGESTION_MIN` | `250` | Seuil d'estomac au-delà duquel l'agent dépose des déchets. |
| `DIFFUSION_RATE` | `2/3` | Fraction de l'énergie déposée répartie aux cases voisines. |
| `SEUIL_DIFFUSION` | `250` | Sous ce seuil, plus de fragmentation (éviter les miettes). |
| `WIDTH` / `HEIGHT` | `1280` / `720` | *(options, hors PARAMS)* : taille du monde, lues si présentes dans le dict. |
| `N_ITER` | `1` | *(options, hors PARAMS)* : pas logiques par image (accélère le temps simulé). |

Pour tester une variation de paramètres en une commande :
```bash
.venv/bin/python Environnement.py --headless --seed 3 \
    --param NUM_AGENTS=300 --param PROBA_INSERTION=0.6 --param MAX_ITERATIONS=5000 \
    --outdir essai_insertion_06
```

---

## 6. Contenu d'une simulation (`simulations/<nom>/`)

| Fichier | Contenu |
|---|---|
| `parameters.json` | Paramètres effectivement utilisés (PARAMS + surcharges). |
| `stats.json` | Historique des stats (tous les 100 pas) : population, taille moyenne des réseaux, énergie moyenne, **énergie globale**, + totaux finaux. |
| `evolution.db` | Base SQLite de l'historique des cerveaux (voir §7). |
| `simulation_summary.png` | Résumé graphique en fin de simulation. |
| `exports/` | *(au-delà de 50 000 pas)* exports des cerveaux toutes les 50 000 pas (CSV + graphes) via `ExportData.py` — alimente `Analyse.py`. |
| `campaign.json` | *(lanceur)* provenance : campagne, run, seed, surcharges. |
| `run.log` | *(lanceur)* sortie du processus. |

---

## 7. Historique des cerveaux (base SQLite « git-like »)

**`evolution.db`** contient trois tables :

- `agents` : `id_agent`, `id_parent`, `iteration_creation` — l'arbre généalogique.
- `networks` : `id_agent`, `iteration`, `network_data` (JSON complet : états, poids,
  identifiants des nœuds), `network_hash`, `is_snapshot`.
- `commits` : `id_agent`, `iteration`, `diff_data` (**delta jsonpatch** par rapport au
  cerveau du parent), `id_parent_commit`, `message`.

Écritures :
- **À chaque naissance** : le réseau de l'enfant est enregistré (JSON complet dans
  `networks` + **delta** par rapport au parent dans `commits`) — une mutation = un commit.
- **Toutes les 50 000 pas** : *snapshot* périodique de chaque agent vivant
  (`is_snapshot=TRUE`).

Lecture / reconstruction — depuis n'importe quel script Python :
```python
from evolution_tracker import rebuild_network, close_all_connections

# Cerveau de l'agent 10 tel qu'il existait à l'itération 60 000 :
network = rebuild_network("simulations/<nom>/evolution.db", 10, 60000)
# -> {"states": [...], "weights": [[...]]}  (taille du réseau = len(states))
close_all_connections()
```
`rebuild_network` repart du dernier snapshot ≤ l'itération cible et **rejoue les
deltas** (jsonpatch) — c'est le chemin git. Sans snapshot, il lit directement le réseau
complet. (Exemple complet : `analyse_evolution.py`.)

---

## 8. Lanceur de campagnes parallèles (`launcher.py`)

Le vrai usage recherche = **beaucoup de simulations** (seeds / paramètres différents).
La machine a 32 cœurs : le lanceur en profite.

```bash
.venv/bin/python launcher.py            # panneau web sur http://127.0.0.1:8765
.venv/bin/python launcher.py --port 9000
```

**Interface web** (aucune dépendance externe) :
- Formulaire : étiquette, nb de runs, graine de base (run *i* = base + *i*), pas max
  (`0` = le paramètre `MAX_ITERATIONS` s'applique), parallélisme max (vide = auto),
  surcharges de paramètres (une `CLE=VALEUR` par ligne).
- Suivi en temps réel par campagne / par run : état, durée, nombre de pas, population
  finale, dossier de sortie, courbe de population, fin de log.
- Bouton **Arrêter** : SIGTERM puis **SIGKILL** après 3 s (nécessaire : voir §11),
  les runs non démarrés passent en `skipped`.

**API** (pour scripter) :
```
GET  /api/state                        → tout l'état (campagnes, runs)
POST /api/campaign                     → créer : {label, n_runs, seed_base, max_steps,
                                           max_parallel, params: "CLE=VALEUR\n..."}
POST /api/campaign/<id>/stop           → arrêter une campagne
GET  /api/run/<camp_id>/<run_id>/log   → fin du log (?lines=N)
GET  /api/run/<camp_id>/<run_id>/curve → courbe population {steps:[...], pop:[...]}
```

Chaque run est un **sous-processus indépendant** (`Environnement.py --headless ...`)
dans son propre dossier `simulations/<campagne>_seed<S>_runNN/` — donc aucune
interférence, et les 32 cœurs sont exploités pour les **campagnes** (pas pour une
simulation unique).

---

## 9. Post-analyse

| Outil | Usage |
|---|---|
| `Analyse.py` | Pipeline graphique sur les **exports** (CSV `agents_save_*.csv` produits par `ExportData.py` toutes les 50 000 pas) : `python Analyse.py` — parcourt `simulations/`, génère `1_temporel/`, `2_spatial/`, `3_moyennes_barplot/`. |
| `analyse_evolution.py` | Exemple de reconstruction de cerveau à une itération (adapter `db_path`, `agent_id`, `target_iteration`). |
| `genealogy_viewerv2.py` | Visualiseur Tkinter de la généalogie (v2 recommandée). |
| `lignéfixv2.py` | Généalogie networkx sur `evolution.db` **du dossier courant** (copier la base ou lancer depuis le bon dossier). |

Les scripts « ad hoc » (`analyse_evolution.py`, `lignéfixv2.py`) ont le chemin de la
base **codé en dur en relatif** : les lancer depuis le dossier qui contient la
`evolution.db` visée.

---

## 10. Invariant de contrôle et vérifications

> **L'énergie globale est strictement conservée à chaque pas.**
> `Σ (agent.énergie + agent.estomach) + Σ nourriture.énergie` reste constant
> (155 000.0 sur les runs de référence). Toute modification du simulateur qui fait
> dériver cette valeur a créé un fuit/bug d'énergie.

Vérifications à faire **après toute modification** (et avant de dire « c'est bon ») :
1. `py_compile` des fichiers modifiés.
2. Run headless court (`--max-steps 2000 --seed 7`) : pas de crash, stats cohérentes.
3. Comparaison de l'énergie globale début/fin du run (dérive attendue : **0**).
4. Pour les fonctions pures (`SbNetwork`, `DiffusionGrid`, `SpatialGrid`) : comparer
   les sorties avec une seed fixée, avant/après.

---

## 11. Pièges connus

- **Headless : le SIGTERM ne tue pas la simulation.** pygame/SDL installe un handler qui
  transforme le SIGTERM en événement `pygame.QUIT` ; le QUIT est désormais ignoré en
  headless (fix 2026-10-16 — avant, les runs headless mouraient **aléatoirement**,
  parfois après 30 pas, avec des stats tronquées et un exit 0). Conséquence pratique :
  pour tuer un run headless il faut **SIGKILL** — le lanceur l'escadre automatiquement
  après 3 s de SIGTERM. En mode fenêtre, le QUIT (bouton croix) arrête normalement.
- **Reproductibilité** : avec la même `SEED`, un run headless est reproductible ;
  attention, tout changement d'ordre d'itération ou de graine décale la trajectoire.
- **Collisions de dossiers** : le nom de dossier par défaut est l'horodatage à la
  seconde ; deux lancements manuels dans la même seconde écriraient dans le même
  dossier. Le lanceur contourne en passant toujours `--outdir` explicite.
- **Dossiers/doublons** : `save_network_state.py`, `SpatialGridAgent.py`,
  `simulation_setup.py` sont morts (voir §2) — ne pas les « réparer », les supprimer
  quand le besoin se fera sentir.
- **Performance** (mesurée, 2026-10-16) : ~8 ms/pas à 150 agents (dont ~7 ms = le
  graphe matplotlib de suivi tous les 100 pas, **même en headless**), ~62 ms/pas à
  500, ~1000 ms/pas à 2000 (croissance super-linéaire : à haute population, ~47 % du
  temps = commits SQLite (un par naissance), ~24 % = perception/voisins). Gagnera le
  plus : désactiver les graphes en headless + écrire la base en asynchrone.

---

## 12. Exemples de sessions

```bash
# Run interactif classique
.venv/bin/python Environnement.py

# Benchmark reproductible
.venv/bin/python Environnement.py --headless --max-steps 2000 --seed 7

# Campagne manuelle : 10 seeds, 3000 pas, 200 agents
for s in $(seq 1 10); do
  .venv/bin/python Environnement.py --headless --seed $s --max-steps 3000 \
      --param NUM_AGENTS=200 --outdir campagneA_seed$s &
done; wait

# Campagne via le panneau web
.venv/bin/python launcher.py
# → http://127.0.0.1:8765
```
