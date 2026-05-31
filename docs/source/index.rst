.. Documentation principale — Prévision des Prix de l'Or au Maroc

Prévision des Prix de l'Or au Maroc
=====================================

.. image:: https://img.shields.io/badge/python-3.10%2B-blue
   :alt: Python 3.10+

.. image:: https://img.shields.io/badge/statut-Recherche-orange
   :alt: Recherche académique

**Système de prévision mensuelle des prix de l'or en Dirham marocain (MAD) — niveau académique.**

Ce projet combine l'économétrie classique, le machine learning et le deep learning pour
prévoir le prix de l'or au Maroc à fréquence mensuelle, d'août 2000 à décembre 2027.
Le pipeline est entièrement reproductible, prévient strictement toute fuite de données,
et produit des figures statiques, des tableaux et des prévisions CSV — sans tableau de bord ni interface web.

.. note::

   **Auteur :** Djibril SALL — École Nationale Supérieure d'Arts et Métiers (ENSAM)

   **Encadrant :** Pr. Tawfik Masrour

----

Démarrage rapide
-----------------

.. code-block:: bash

   # Cloner et installer
   python -m venv .venv
   .venv\Scripts\activate       # Windows
   # source .venv/bin/activate  # Linux / macOS
   pip install -r requirements.txt

   # Lancer le pipeline complet
   python main.py

   # Optionnel : télécharger les indicateurs macro via yfinance
   python main.py --build-macro

   # Régénérer uniquement les prévisions (sans le backtest)
   python main.py --forecast-only

Après une exécution réussie, les résultats sont écrits dans :

* ``results/figures/``   — EDA, diagnostics, réel vs prédit, résidus
* ``results/tables/``    — classements des modèles, intervalles de confiance bootstrap, stabilité par fold
* ``results/forecasts/`` — prévisions futures jusqu'en décembre 2027 (CSV)
* ``results/models/``    — métadonnées JSON du pipeline, rapport d'analyse

----

.. toctree::
   :maxdepth: 2
   :caption: Présentation du projet

   presentation

.. toctree::
   :maxdepth: 2
   :caption: Guide utilisateur

   datasets
   preprocessing
   feature_engineering
   forecasting
   results

.. toctree::
   :maxdepth: 2
   :caption: Référence

   metrics/index
   models/index
   conclusion

----

Résultats clés
--------------

.. list-table::
   :header-rows: 1
   :widths: 26 14 14 12 16 12

   * - Modèle
     - RMSE moy.
     - MAE moy.
     - MAPE %
     - Dir. Acc. %
     - Biais
   * - **SARIMAX ★**
     - **1 478**
     - **1 230**
     - **6,49**
     - 60,9
     - −797
   * - ARIMA
     - 1 536
     - 1 296
     - 6,92
     - 55,5
     - −865
   * - Hybride ARIMA+XGB
     - 1 544
     - 1 319
     - 7,26
     - 58,2
     - −772
   * - SARIMA
     - 1 616
     - 1 363
     - 7,42
     - 54,5
     - −927
   * - XGBoost
     - 1 759
     - 1 525
     - 7,92
     - 60,9
     - −1 438
   * - LSTM (univarié)
     - 1 979
     - 1 741
     - 9,10
     - 47,3
     - −1 573
   * - Prophet
     - 2 406
     - 2 026
     - 12,30
     - 46,4
     - −1 100

*Métriques moyennées sur 10 folds en fenêtre expansive (2015–2025, horizon de test de 12 mois par fold).*

**Modèle recommandé en production :** ``Ensemble_Weighted``
(combinaison pondérée par l'inverse du RMSE : SARIMAX + XGBoost + Hybride ARIMA-XGBoost).
