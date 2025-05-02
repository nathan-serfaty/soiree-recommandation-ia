# 🎉 Soirée Recommandation — Application IA

## 📌 Objectif du projet

Ce projet vise à recommander dynamiquement des lieux de sortie (bar, restaurant, boîte de nuit, etc.) en fonction du profil de l'utilisateur (âge, genre, ville, préférence) à l'aide d'un modèle de machine learning.

L'utilisateur remplit un court formulaire et reçoit immédiatement une suggestion personnalisée, accompagnée d'une liste des lieux les mieux notés correspondant au type prédit.

---

## 🧾 1. Données utilisées (8 points)

- **Type de données :**  
  Données synthétiques générées à l’aide de la librairie [Faker](https://faker.readthedocs.io/en/master/), enrichies par une logique conditionnelle pour créer un lien réaliste entre préférences utilisateur et type de lieu recommandé.

- **Colonnes :**  
  `age`, `genre`, `ville`, `preference` → `type`, `nom_lieu`, `note_moyenne`

- **Source :**  
  Données simulées (aucune donnée personnelle réelle).

- **Prétraitement :**  
  Les variables catégorielles (`genre`, `ville`, `preference`, `type`) sont encodées avec `LabelEncoder`, sauvegardées via `joblib`.

- **Utilité :**  
  Ces données permettent d'entraîner un modèle de classification pour **prédire un type de lieu pertinent** à recommander selon le profil de l'utilisateur.

---

## 🤖 2. Méthodes de machine learning utilisées (6 points)

- **Modèle principal :**  
  `XGBClassifier` (XGBoost) — Classificateur puissant et robuste pour les tâches multi-classes.

- **Pourquoi ce choix ?**
  - Très performant même sur données simulées
  - Gère bien les colonnes numériques encodées
  - Facile à intégrer dans un pipeline scikit-learn
  - Compatible avec analyse de performance (`metrics.txt` généré)

- **Méthodologie :**
  - Séparation des features (`X`) et de la cible (`y`)
  - Entraînement avec `model.fit(X, y)`
  - Sauvegarde du modèle dans `model/model.joblib`

---

## ⚙️ 3. Lancer le projet en local

### Prérequis

- Python 3.8+ recommandé
- `pip`, `virtualenv` ou `venv`

### Étapes

```bash
# 1. Cloner le repo
git clone <URL_DU_REPO>
cd soirée_ml/project_root

# 2. Créer un environnement virtuel
python3 -m venv env
source env/bin/activate

# 3. Installer les dépendances
pip install -r requirements.txt

# 4. Entraîner le modèle (optionnel si déjà présent)
python train_model.py

# 5. Lancer l'application web
python -m app.main
