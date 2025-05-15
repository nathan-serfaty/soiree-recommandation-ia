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

### 📦 Installation

```bash
# 1. Cloner le projet
git clone <URL_DU_REPO>
cd soirée_ml/project_root

# 2. Créer un environnement virtuel
python3 -m venv venv
source venv/bin/activate

# 3. Installer les dépendances
pip install -r requirements.txt
# (optionnel car déja pré train dans le repo) Entraîner le modèle ML
python train_model.py

# (optionnel car déja pré train dans le repo) Entraîner l’encodage pour la similarité
python train_similarity.py

# Lancer le serveur Flask
python -m app.main

# Structure du projet:

project_root/
├── app/
│   ├── main.py               # Serveur Flask
│   ├── model.py              # Prédiction générale (ML)
│   ├── similarity.py         # Matching vectoriel
│   ├── templates/
│   │   └── index.html        # Interface web
├── data/
│   └── dataset_sorties_500k.csv
├── model/                    # Modèles, encoders, json exportés
│   ├── model.joblib
│   ├── metrics.json
│   ├── resultats_du_jour.json
├── static/                  # Fichiers téléchargeables (.csv)
│   └── recommandations_du_jour.csv
