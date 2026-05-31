.. _presentation:

Présentation du projet
======================

Contexte et motivation
-----------------------

L'or est universellement considéré comme une valeur refuge et une protection contre
l'inflation. Son prix sur les marchés internationaux (USD par once troy) a progressé de
plus de **750 %** entre 2000 et début 2026. Au Maroc, cependant, le prix effectivement
payé par les consommateurs et les négociants n'est pas le cours spot en USD, mais son
**équivalent en MAD** :

.. math::

   \text{gold\_price\_mad} = \text{gold\_price\_usd} \times \text{usd\_mad}

Cette double exposition — au marché mondial des matières premières **et** au taux de change
USD/MAD — crée un problème de prévision spécifique au marché marocain qui n'a pas encore
fait l'objet d'un pipeline académique dédié.

Pourquoi le Maroc ?
~~~~~~~~~~~~~~~~~~~

Le Maroc présente trois caractéristiques distinctives qui rendent la prévision du prix de
l'or particulièrement intéressante :

1. **Cycles de demande socio-culturels** — Le Ramadan, l'Aïd Al-Fitr, l'Aïd Al-Adha,
   la saison des mariages (avril–septembre) et la saison de retour des MRE (Marocains
   Résidant à l'Étranger) génèrent des pics mensuels prévisibles de la demande en
   bijouterie qui se répercutent sur les prix locaux de l'or.

2. **Sensibilité au taux de change** — Le MAD est géré par rapport à un panier de devises
   dominé par l'EUR et l'USD. Un dollar fort élève simultanément le cours USD de l'or et
   affaiblit le MAD, créant des effets composés sur ``gold_price_mad``.

3. **Contexte de politique macro** — Les décisions de taux de Bank Al-Maghrib (BAM) et
   l'inflation domestique interagissent avec les dynamiques de coûts à l'importation
   d'une manière différente de celle des économies avancées.

Objectifs de recherche
-----------------------

Le projet répond à quatre questions :

* Les modèles économétriques classiques (ARIMA, SARIMA, SARIMAX) surpassent-ils les
  références en machine learning (XGBoost, LSTM) pour les prix mensuels de l'or au Maroc ?
* Les indicateurs d'événements socio-culturels marocains apportent-ils un pouvoir prédictif
  statistiquement significatif lorsqu'ils sont ajoutés comme régresseurs exogènes ?
* Quelle est la stratégie d'ensemble optimale pour une prévision de production à 12–24 mois ?
* Comment la stabilisation de tendance (amortissement, retour à la moyenne) améliore-t-elle
  la plausibilité économique des prévisions à long horizon ?

Cadre académique
-----------------

Ce projet a été réalisé à l'**École Nationale Supérieure d'Arts et Métiers (ENSAM)**
sous la direction de **Pr. Tawfik Masrour**.

* **Auteur** : Djibril SALL
* **Niveau** : Élève-Ingénieur
* **Portée** : Recherche académique / pipeline reproductible

Structure du projet
--------------------

.. code-block:: text

   projet/
   ├── data/
   │   ├── gold_prices.csv
   │   ├── usd_mad.csv
   │   ├── moroccan_events.csv
   │   └── macro_indicators.csv
   ├── notebooks/
   │   └── 01_exploratory_analysis.ipynb
   ├── src/
   │   ├── preprocessing/        # chargement des données, nettoyage, préprocesseur par fold
   │   ├── feature_engineering/  # features lag/roll, encodage événements, exogènes futurs
   │   ├── models/               # ARIMA, SARIMA, SARIMAX, Prophet, XGBoost, LSTM, Ensemble
   │   ├── evaluation/           # backtesting, métriques, IC bootstrap, stabilité
   │   ├── forecasting/          # orchestrateur pipeline, calibration, stabilisation tendance
   │   └── utils/                # config, vérifications d'alignement, graphiques, reproductibilité
   ├── results/
   │   ├── figures/
   │   ├── tables/
   │   ├── forecasts/
   │   └── models/
   ├── tests/
   │   ├── test_leakage_detection.py
   │   ├── test_backtest_integrity.py
   │   └── test_xgboost_alignment.py
   ├── main.py
   ├── requirements.txt
   └── README.md

Vue d'ensemble du pipeline
---------------------------

Le pipeline s'exécute en une seule commande (``python main.py``) et enchaîne les étapes
suivantes :

.. list-table::
   :header-rows: 1
   :widths: 5 25 45 25

   * - #
     - Étape
     - Description
     - Sortie
   * - 1
     - **Chargement des données**
     - Fusion or, FX, événements et macro ; calcul de ``gold_price_mad`` ; pas d'imputation globale
     - ``DataFrame`` brut
   * - 2
     - **Diagnostics EDA**
     - Matrice de corrélation, décomposition saisonnière, test ADF, ACF/PACF, graphiques d'impact événementiel
     - ``results/figures/``
   * - 3
     - **Détection de régimes**
     - Points de rupture par z-score glissant (57 détectés), régime de volatilité, force de tendance
     - ``regime_adf_segments.csv``
   * - 4
     - **Backtest glissant**
     - 10 folds en fenêtre expansive ; prétraitement par fold ; tous modèles + références naïves
     - ``rolling_cv_by_fold.csv``
   * - 5
     - **Comparaison des modèles**
     - RMSE/MAE/MAPE/R²/biais/précision directionnelle ; IC bootstrap à 95 % (500 échantillons)
     - ``model_ranking_backtest.csv``, ``metrics_bootstrap_ci.csv``
   * - 6
     - **Log vs niveau**
     - Backtest ARIMA/Prophet/XGBoost en espace log ; choix de la meilleure transformation par modèle
     - ``log_vs_level_comparison.csv``
   * - 7
     - **Prévision future**
     - Réentraînement sur l'échantillon complet ; amortissement de tendance φ=0,92 ; correction de biais ; Ensemble_Weighted
     - ``future_forecasts_to_2027_12.csv``
   * - 8
     - **Intervalles probabilistes**
     - Éventail bootstrap sur les résidus (IC à 95 %)
     - ``forecast_intervals_2027_12.csv``
   * - 9
     - **Rapport**
     - Rapport d'analyse Markdown + métadonnées JSON du pipeline
     - ``analysis_report.md``, ``pipeline_run_metadata.json``

Reproductibilité
-----------------

Une graine aléatoire fixe (``RANDOM_SEED = 42``) est appliquée à NumPy, au module
``random`` de Python et à TensorFlow au début de chaque exécution via
``src/utils/reproducibility.py``.

L'empreinte complète des versions de packages est sauvegardée dans
``results/models/pipeline_run_metadata.json`` à chaque exécution.

Exécutez la suite de tests pour vérifier l'intégrité du pipeline avant toute expérience :

.. code-block:: bash

   pytest tests/ -v
