.. _results:

Résultats
=========

Cette section présente les résultats quantitatifs du pipeline : performances du backtest
glissant, analyse de régimes, intervalles de confiance bootstrap et prévisions futures.

Tous les fichiers de sortie sont écrits dans ``results/``.

----

Résumé du backtest glissant
-----------------------------

Le pipeline évalue tous les modèles sur **10 folds en fenêtre expansive** (2015–2025).
Chaque fold utilise un ``FoldPreprocessor`` ajusté uniquement sur la moitié
d'entraînement.

Métriques moyennes sur les folds
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 26 12 12 12 14 12 8

   * - Modèle
     - RMSE
     - MAE
     - MAPE %
     - Dir. Acc. %
     - Biais
     - R²
   * - **SARIMAX ★**
     - **1 478**
     - **1 230**
     - **6,49**
     - 60,9
     - −797
     - −1,46
   * - Référence Drift
     - 1 487
     - 1 260
     - 6,83
     - 60,9
     - −756
     - −1,95
   * - ARIMA
     - 1 536
     - 1 296
     - 6,92
     - 55,5
     - −865
     - −1,87
   * - Hybride ARIMA+XGB
     - 1 544
     - 1 319
     - 7,26
     - 58,2
     - −772
     - −2,41
   * - SARIMA
     - 1 616
     - 1 363
     - 7,42
     - 54,5
     - −927
     - −2,63
   * - Référence DernièreValeur
     - 1 683
     - 1 430
     - 7,66
     - 0,0
     - −1 098
     - −2,46
   * - XGBoost
     - 1 759
     - 1 525
     - 7,92
     - 60,9
     - −1 438
     - −2,36
   * - LSTM (univarié)
     - 1 979
     - 1 741
     - 9,10
     - 47,3
     - −1 573
     - −3,88
   * - Prophet
     - 2 406
     - 2 026
     - 12,30
     - 46,4
     - −1 100
     - −16,87
   * - Référence MoyenneEntraînement
     - 7 954
     - 7 894
     - 44,78
     - 0,0
     - −7 894
     - −114,1

.. note::

   Le **R² du backtest glissant** est négatif pour tous les modèles car il est calculé
   par rapport à la moyenne hors échantillon — un benchmark non trivial pour une série
   fortement croissante. Le **R² bootstrap** ci-dessous (calculé sur les prévisions OOS
   regroupées) est positif et compris entre 0,71 et 0,87, reflétant un fort pouvoir
   explicatif hors échantillon.

Sauvegardé dans : ``results/tables/model_ranking_backtest.csv``

----

Intervalles de confiance bootstrap (OOS regroupé)
---------------------------------------------------

Bootstrap à 500 échantillons sur les prévisions hors échantillon regroupées (10 folds
concaténés) :

.. list-table::
   :header-rows: 1
   :widths: 30 30 18 28

   * - Modèle
     - RMSE [IC 95 %]
     - R²
     - R² [IC 95 %]
   * - **SARIMAX**
     - 1 881 [1 506 – 2 272]
     - **0,859**
     - [0,827 – 0,892]
   * - Hybride ARIMA+XGB
     - 1 821 [1 506 – 2 129]
     - 0,868
     - [0,840 – 0,894]
   * - ARIMA
     - 1 905 [1 544 – 2 289]
     - 0,856
     - [0,824 – 0,887]
   * - SARIMA
     - 1 938 [1 587 – 2 315]
     - 0,851
     - [0,818 – 0,882]
   * - XGBoost
     - 2 270 [1 818 – 2 724]
     - 0,795
     - [0,749 – 0,837]
   * - LSTM
     - 2 508 [2 118 – 2 938]
     - 0,750
     - [0,696 – 0,789]
   * - Prophet
     - 2 693 [2 324 – 3 078]
     - 0,712
     - [0,612 – 0,774]

Sauvegardé dans : ``results/tables/metrics_bootstrap_ci.csv``

----

Comparaison log vs niveau pour la variable cible
--------------------------------------------------

Trois modèles ont été réévalués avec une cible transformée en logarithme :

.. list-table::
   :header-rows: 1
   :widths: 18 18 18 18 18 10

   * - Modèle
     - RMSE (niveau)
     - RMSE (log)
     - MAE (niveau)
     - MAE (log)
     - Meilleur
   * - ARIMA
     - 1 536
     - **1 368**
     - 1 296
     - **1 131**
     - Log ✓
   * - Prophet
     - 2 406
     - **2 292**
     - 2 026
     - **1 993**
     - Log ✓
   * - XGBoost
     - **1 759**
     - 16 628
     - **1 525**
     - 16 599
     - Niveau ✓

Résultat clé : XGBoost en espace log échoue catastrophiquement (RMSE × 9,5) car la
retransformation récursive amplifie exponentiellement les petites erreurs de prévision.

Sauvegardé dans : ``results/tables/log_vs_level_comparison.csv``

----

Analyse des régimes
---------------------

Détection des points de rupture
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

La fonction ``run_regime_analysis()`` (``src/analysis/regime_detection.py``) applique
un détecteur de z-score glissant (fenêtre = 24 mois, seuil = 2,0 écarts-types) :

* **57 points de rupture structurels** détectés sur la série complète.
* Principaux regroupements : hausse de l'or 2005–2006, crise financière 2008, record
  historique 2011, correction 2013, consolidation 2016, COVID 2019–2020, choc
  matières premières 2022, nouveau marché haussier 2024–2026.

Régime actuel (février 2026)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :widths: 40 20 40

   * - Indicateur
     - Valeur
     - Interprétation
   * - Régime de volatilité (0=faible, 1=élevée)
     - **1,0**
     - Volatilité élevée — les variations mensuelles récentes dépassent la médiane historique
   * - Force de tendance (12 mois)
     - **0,578**
     - Tendance modérée à forte à la hausse

Test ADF par segment de régime
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Les tests ADF sur les 10 plus grands segments entre points de rupture consécutifs
confirment que la série est non stationnaire dans chaque sous-période (p-valeurs > 0,05
dans tous les segments d'au moins 12 observations), validant le choix universel de d = 1.

Sauvegardé dans : ``results/tables/regime_adf_segments.csv``

----

Stabilité des modèles par fold
--------------------------------

``src/analysis/stability_analysis.py`` exporte le RMSE, MAE, R² et biais par fold :

**Plage de RMSE par fold (SARIMAX) :**

* Meilleur fold : ~400 MAD (période stable 2016–2018)
* Pire fold : ~3 100 MAD (période volatile 2024–2025)
* Écart-type : 1 165 MAD

L'écart-type élevé reflète la nature hétéroscédastique de la série — la volatilité du
prix de l'or augmente proportionnellement au niveau des prix.

Sauvegardé dans : ``results/tables/rolling_cv_by_fold.csv``,
``results/tables/model_stability_summary.csv``

----

Prévisions futures 2026–2027
------------------------------

Tous les modèles sont réentraînés sur l'**échantillon complet** (août 2000 – février 2026)
et prévoient jusqu'en décembre 2027 (22 mois).

.. list-table::
   :header-rows: 1
   :widths: 14 14 14 14 14 14 10 10

   * - Date
     - Ensemble
     - SARIMAX
     - XGBoost
     - ARIMA
     - Prophet
     - LSTM
     - Hybride
   * - Mars 2026
     - 45 870
     - 45 929
     - 43 980
     - 45 205
     - 34 840
     - 32 333
     - 44 930
   * - Juin 2026
     - 47 261
     - 49 253
     - 43 889
     - 46 142
     - 37 258
     - 33 977
     - 46 100
   * - Sep. 2026
     - 48 569
     - 51 672
     - 43 885
     - 46 548
     - 38 757
     - 34 454
     - 47 050
   * - Déc. 2026
     - 49 592
     - 53 635
     - 43 584
     - 46 559
     - 40 127
     - 33 750
     - 47 780
   * - Mars 2027
     - 50 351
     - 54 479
     - 43 586
     - 46 240
     - 32 706
     - 32 254
     - 48 200
   * - Juin 2027
     - 50 684
     - 54 266
     - 43 576
     - 45 679
     - 38 638
     - 32 085
     - 48 310
   * - Sep. 2027
     - 50 862
     - 53 386
     - 43 584
     - 44 958
     - 43 798
     - 31 568
     - 48 290
   * - Déc. 2027
     - **50 906**
     - 52 182
     - 43 588
     - 44 151
     - 41 828
     - 30 974
     - 48 140

*Toutes les valeurs en MAD par once troy.*

Observations principales :

* **Ensemble_Weighted** converge en douceur vers ~51 000 MAD en décembre 2027, avec
  un taux de croissance décélérant — une trajectoire économiquement plausible grâce
  à l'amortissement de tendance.
* **SARIMAX** atteint un pic vers mars 2027 (~54 500 MAD) puis revient graduellement
  vers 52 000 MAD sous l'effet de la composante de retour à la moyenne.
* **XGBoost** est le modèle le plus conservateur, se maintenant près de 43 500 MAD
  tout au long de l'horizon. C'est cohérent avec sa conception récursive.
* **Prophet** présente la plus grande variance inter-périodes, reflétant son incapacité
  à modéliser le régime actuel de forte volatilité.
* **LSTM** décline sur l'horizon de prévision — un artefact connu des LSTM récursifs
  univariés lorsque la série revient à la moyenne par rapport à la distribution
  d'entraînement.

Sauvegardé dans : ``results/forecasts/future_forecasts_to_2027_12.csv``

Intervalles de prévision (95 %)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Intervalles bootstrap sur les résidus pour SARIMAX :

* Mars 2026 : 45 929 MAD [bande étroite ≈ ±0,03]
* Déc. 2027 : 52 182 MAD [bande ≈ ±8 MAD]

.. note::

   Les intervalles bootstrap étroits s'expliquent par le fait que le bootstrap sur les
   résidus échantillonne à partir des résidus ARIMA en échantillon, qui ont une faible
   variance par rapport au régime récent. L'incertitude prédictive réelle à 22 mois
   est substantiellement plus grande et doit être interprétée avec prudence.

Sauvegardé dans : ``results/forecasts/forecast_intervals_2027_12.csv``
