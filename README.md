# 🎉 Soirée Recommandation — Application IA

Bienvenue dans cette application de **recommandation intelligente de lieux de sortie** (bars, restaurants, boîtes de nuit, etc.), basée sur un **modèle de machine learning** et un système de **matching vectoriel** personnalisé.

---

## 📌 Objectif du projet

Permettre à un utilisateur d’obtenir, via un formulaire web simple :
- une **recommandation générale** (via modèle ML : XGBoost, RandomForest…)
- une **recommandation précise et personnalisée** (via recherche par similarité)

Chaque prédiction renvoie :
- Le type de lieu prédit (ex: `bar`, `restaurant`, etc.)
- Les **5 meilleurs lieux notés** dans ce type
- Les **métriques du modèle**
- Un lien pour **télécharger toutes les recommandations** (.csv)
- Et un **résumé JSON complet** accessible à `/download`

---

## 🧾 1. Données utilisées

- **Fichier :** `data/dataset_sorties_500k.csv`
- **Colonnes :**
  - `age`, `genre`, `ville`, `preference`
  - → `type` (cible), `nom_lieu`, `note_moyenne`
- **Prétraitement :**
  - Encodage avec `LabelEncoder`
  - Sauvegardes des encodeurs dans `model/`
- **Ajouts pour similarité :**
  - `moment`, `ambiance`, `budget`, `avec` (ajoutés aléatoirement pour enrichir le matching)
- **Stockage vectoriel :**
  - Vecteurs encodés sauvegardés avec `joblib` pour recherche vectorielle (cosine similarity)

---

## 🤖 2. Modèles utilisés

### 🔹 Classification (Recommandation Générale)

- **Modèles disponibles :** `XGBoost`, `RandomForest`
- **Fonctionnement :**
  - Prédiction du `type` de lieu à recommander
  - Renvoi des lieux correspondants triés par `note_moyenne`
  - Sauvegarde CSV dans `static/recommandations_du_jour.csv`
  - Sauvegarde JSON complète dans `model/resultats_du_jour.json`

### 🔹 Similarité (Recommandation Personnalisée)

- **Méthode :** `cosine_similarity` sur vecteurs encodés (`OneHotEncoder`)
- **Objectif :** proposer les lieux les plus proches du **profil utilisateur enrichi**
- **Colonnes utilisées :**
  - `genre`, `ville`, `moment`, `ambiance`, `budget`, `avec`

---

## 🖥️ 3. Interface Web

- **Technologie :** Flask + HTML (template `index.html`)
- Deux formulaires :
  - 🔵 Recommandation Générale (modèle ML)
  - 🟣 Recommandation Précise (matching vectoriel)
- Résultat affiché dynamiquement dans l’interface
- 📥 Lien direct vers les fichiers recommandés
- 🔄 Résultats accessibles via :
  - `/predict` (POST)
  - `/similar` (POST)
  - `/download` (GET) → retourne un JSON :  
    ```json
    {
      "type": "bar",
      "model": "xgboost",
      "lieux": [...],
      "download_url": "/static/recommandations_du_jour.csv"
    }
    ```

---

## ⚙️ 4. Lancer le projet localement

### 🔧 Prérequis

- Python 3.8+
- pip

📦 Installation
# 1. Cloner le projet
git clone <URL_DU_REPO>
cd soiree_ml/project_root

# 2. Créer un environnement virtuel
python3 -m venv venv
source venv/bin/activate

# 3. Installer les dépendances
pip install -r requirements.txt

# (optionnel) Entraîner le modèle de machine learning
python train_model.py

# (optionnel) Entraîner l’encodage pour la similarité
python train_similarity.py

# Lancer le serveur Flask
python -m app.main


📁 Structure du projet
L’arborescence du projet est organisée comme suit :

project_root/
├── app/                            # Code principal de l'application Flask
│   ├── main.py                     # Script principal (serveur Flask)
│   ├── model.py                    # Prédictions générales (ML)
│   ├── similarity.py               # Algorithme de similarité vectorielle
│   └── templates/
│       └── index.html              # Interface utilisateur (HTML)
│
├── data/                           # Données sources
│   └── dataset_sorties_500k.csv    # Dataset principal
│
├── model/                          # Fichiers liés au modèle ML
│   ├── model.joblib                # Modèle entraîné
│   ├── metrics.json                # Métriques de performance
│   ├── resultats_du_jour.json      # Résultats prédits
│
├── static/                         # Fichiers statiques générés
│   └── recommandations_du_jour.csv# Fichier de recommandations à télécharger
│
├── requirements.txt                # Dépendances Python
├── train_model.py                  # Script d'entraînement du modèle
├── train_similarity.py             # Script de préparation pour la similarité
└── README.md                       # Documentation du projet

