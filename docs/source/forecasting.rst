.. _forecasting:

Prévision
=========

Cette section couvre les sept modèles individuels, les stratégies hybrides et d'ensemble,
la stabilisation de tendance, la correction de biais et l'estimation des intervalles
probabilistes.

Tout le code des modèles se trouve dans ``src/models/``. Les utilitaires de calibration
et de post-traitement sont dans ``src/forecasting/``.

----

Interface de base
-----------------

Chaque modèle hérite de ``BaseForecaster`` (``src/models/base.py``) :

.. code-block:: python

   class BaseForecaster(ABC):
       name: str

       def fit(self, y: pd.Series,
               exog: pd.DataFrame | None = None) -> "BaseForecaster":
           ...

       def predict(self, steps: int,
                   exog: pd.DataFrame | None = None,
                   index: pd.DatetimeIndex | None = None) -> pd.Series:
           ...

``fit()`` reçoit toujours la **cible d'entraînement** ``y`` et optionnellement le
DataFrame d'entraînement complet ``exog`` (obligatoire pour SARIMAX et XGBoost).
``predict()`` retourne une ``pd.Series`` indexée par le ``DatetimeIndex`` futur demandé.

----

ARIMA — ``arima_model.py``
--------------------------

``ARIMAForecaster`` encapsule ``pmdarima.auto_arima`` avec un pré-contrôle ADF et des
diagnostics ACF/PACF.

**Sélection des ordres :**

.. code-block:: python

   auto_arima(y, seasonal=False, stepwise=True,
              max_p=3, max_q=3, d=None)

``d`` est déterminé automatiquement à partir du test ADF. Étant donné la statistique
ADF de +2,30 et la p-valeur de 0,999, ``d = 1`` est sélectionné dans chaque fold.

**Diagnostics sauvegardés à l'ajustement :**

* ``results/figures/arima_adf.txt`` — statistique ADF et p-valeur
* ``results/figures/arima_acf_pacf.png`` — ACF et PACF jusqu'au lag 24

**Stabilisation de tendance :** appliquée (``STABILIZE_MODELS`` contient ``"ARIMA"``).

**Expérience log-cible :** ARIMA est le meilleur modèle en espace log (RMSE 1 368 vs
1 536 en espace niveau) ; utilisé en espace log dans l'exécution finale de production.

----

SARIMA — ``sarima_model.py``
-----------------------------

``SARIMAForecaster`` étend ``ARIMAForecaster`` avec ``seasonal=True, m=12`` :

.. code-block:: python

   auto_arima(y, seasonal=True, m=12,
              max_P=2, max_Q=2, stepwise=True)

Le graphique ACF confirme une décroissance lente et le PACF montre un seul pic
significatif au lag 1, cohérent avec une structure ARIMA(1,1,0)×(1,1,0)₁₂.

**Stabilisation de tendance :** appliquée.

----

SARIMAX — ``sarimax_model.py``
-------------------------------

``SARIMAXForecaster`` est le meilleur modèle individuel (RMSE moyen 1 478 MAD sur
10 folds).

Il augmente SARIMA avec des régresseurs exogènes sélectionnés dans cet ordre de
priorité :

1. ``usd_mad``
2. Les cinq colonnes d'événements marocains
3. Les colonnes macro disponibles

**Estimations des coefficients (fit final sur l'échantillon complet) :**

.. list-table::
   :header-rows: 1
   :widths: 30 20 50

   * - Paramètre
     - Estimation
     - Interprétation
   * - ``usd_mad``
     - +1 048,5
     - Effet FX dominant : 1 MAD de dépréciation → +1 048 MAD/once
   * - ``eid_aladha``
     - +70,7
     - Pic de demande lors de l'Aïd Al-Adha
   * - ``wedding_season``
     - +48,1
     - Demande de bijoux nuptiaux
   * - ``ramadan``
     - +22,0
     - Achats de bijoux avant l'Aïd
   * - ``dxy_index``
     - −79,2
     - Un dollar fort réduit l'or en termes MAD
   * - ``oil_brent_usd``
     - −17,7
     - Rotation risk-on loin de l'or
   * - ``fed_funds_rate``
     - −0,29
     - Coût d'opportunité de détenir de l'or
   * - ``ma.L1``
     - +0,166
     - Terme de correction résiduelle MA(1)

**IC bootstrap R² (OOS poolé) :** 0,859 [0,827 – 0,892]

**Variables exogènes de test lors du backtesting :** Quand ``STRICT_EVALUATION_MODE = True``,
les valeurs exogènes du fold de test sont tirées des **données observées**, pas extrapolées.
Cela évite de gonfler les performances du backtest avec des connaissances macro futures.

----

Prophet — ``prophet_model.py``
-------------------------------

``ProphetForecaster`` utilise la bibliothèque Prophet de Meta avec :

* ``yearly_seasonality = True``
* Saisonnalité mensuelle personnalisée (ordre de Fourier 5)
* Les cinq colonnes d'événements marocains ajoutées comme régresseurs additifs

**Faiblesses observées lors du backtest :**

Prophet atteint un MAPE moyen de **12,3 %** — presque le double de SARIMAX. Le
principal mode d'échec est son incapacité à suivre les changements brusques de régime
du prix de l'or (choc COVID 2020, pic de matières premières 2022, hausse 2024–2026).
La tendance linéaire par morceaux de Prophet ne peut pas s'adapter assez rapidement
aux frontières des folds.

**Expérience log-cible :** Prophet bénéficie légèrement de l'espace log (RMSE 2 292
vs 2 406) ; utilisé en espace log en production.

----

XGBoost — ``xgboost_model.py``
-------------------------------

``XGBoostForecaster`` effectue une **prévision multi-pas récursive** : à chaque pas
futur, la valeur prédite est ajoutée à l'historique de travail et les features sont
recalculées avant de prédire le pas suivant.

**Hyperparamètres du modèle :**

.. code-block:: python

   XGBRegressor(
       n_estimators=300,
       max_depth=5,
       learning_rate=0.05,
       subsample=0.9,
       colsample_bytree=0.8,
       random_state=42,
   )

**Normalisateur :** ``sklearn.preprocessing.StandardScaler`` ajusté sur les features
d'entraînement uniquement.

**Ensemble de features :** ~85 colonnes (voir :ref:`feature_engineering`). Les features
sont gelées après le fit final sur l'échantillon complet pour garantir un ordre de
colonnes identique à l'inférence.

**Valeurs SHAP :** Si ``shap`` est installé, un graphique de synthèse SHAP est sauvegardé
dans ``results/figures/xgboost_shap_summary.png`` après l'ajustement.

**Faiblesses principales :**

* La dérive récursive s'accumule avec l'horizon — la fiabilité diminue au-delà de
  6 mois.
* La transformation log-cible dégrade significativement les performances (RMSE 16 628
  vs 1 759 en espace niveau) ; XGBoost est toujours entraîné en espace niveau.

----

LSTM — ``lstm_model.py``
-------------------------

``LSTMForecaster`` est un **benchmark d'apprentissage profond univarié** — il utilise
uniquement la série cible (pas de régresseurs exogènes).

**Architecture :**

.. code-block:: text

   LSTM(64, return_sequences=True)
   Dropout(0,2)
   LSTM(32, return_sequences=False)
   Dropout(0,2)
   Dense(16, relu)
   Dense(1)

**Entraînement :**

* Fenêtre de lookback : 12 mois
* Optimiseur : Adam (lr = 0,001)
* Perte : MSE
* Callbacks : ``EarlyStopping(patience=12)``, ``ReduceLROnPlateau(factor=0.5, patience=5)``
* Découpage entraînement/validation : 85 % / 15 % (ordonné dans le temps, pas de mélange)
* Normalisateur : ``MinMaxScaler`` ajusté sur les valeurs d'entraînement uniquement

**Mode de prévision :** Récursif (auto-régressif), comme XGBoost.

**Interprétation :** Le LSTM atteint un R² bootstrap de 0,750 — respectable pour un
modèle univarié sans information macro, mais clairement en dessous des modèles
statistiques avec variables exogènes.

----

Hybride ARIMA + XGBoost — ``ensemble_model.py``
------------------------------------------------

``ARIMAXGBHybrid`` implémente la **modélisation des résidus** :

1. ARIMA est ajusté sur la cible d'entraînement pour capturer la structure linéaire.
2. Les résidus en échantillon ``y - ŷ_ARIMA`` sont calculés.
3. XGBoost est ajusté sur les résidus, en utilisant la matrice complète de features
   comme entrée.
4. Prévision = prévision ponctuelle ARIMA + correction de résidus XGBoost.

Cette conception réduit le biais systématique d'ARIMA (biais moyen −865 → −772 MAD)
tout en conservant la structure de tendance ARIMA interprétable.

----

Ensemble_Weighted (production) — ``ensemble_model.py``
-------------------------------------------------------

``WeightedEnsembleForecaster`` combine trois modèles composants :

* **SARIMAX** (~36 % de pondération)
* **XGBoost** (~31 % de pondération)
* **Hybride ARIMA+XGBoost** (~33 % de pondération)

Les poids sont proportionnels à **1 / RMSE** du backtest glissant :

.. code-block:: python

   from src.forecasting.calibration import inverse_rmse_weights, weighted_ensemble

   poids = inverse_rmse_weights(metrics_bt,
               ["SARIMAX", "XGBoost", "Hybrid_ARIMA_XGBoost"])
   prevision_ensemble = weighted_ensemble(previsions_composants, poids)

L'ensemble n'est pas évalué dans le backtest glissant (il est construit *après*
l'évaluation des modèles individuels). C'est le **modèle recommandé en production**
car il moyenne les modes d'échec spécifiques à chaque modèle selon les régimes
de volatilité.

----

Stabilisation de tendance — ``trend_stabilization.py``
-------------------------------------------------------

Les prévisions à long horizon des modèles de la famille ARIMA tendent à extrapoler
indéfiniment le momentum le plus récent, produisant des trajectoires économiquement
implausibles (ex. +30 % par an). La fonction ``damp_forecast()`` applique trois
corrections :

1. **Décroissance exponentielle de la tendance** — la composante de tendance linéaire
   est multipliée par :math:`\phi^h` où :math:`\phi = 0{,}92` et *h* est l'horizon en mois.
2. **Mélange de retour à la moyenne** — chaque pas est mélangé vers le chemin de
   croissance moyen des 12 derniers mois, pondéré par :math:`(1 - \phi^h)`.
3. **Plafond mois par mois** — la variation prédite est écrêtée à ±4 % du pas précédent
   (``TREND_MAX_MONTHLY_PCT = 0.04``).

**Appliqué à :** ``ARIMA``, ``SARIMA``, ``SARIMAX`` (frozenset ``STABILIZE_MODELS``).

----

Correction de biais — ``calibration.py``
-----------------------------------------

Après le backtest glissant, l'erreur de prévision hors échantillon moyenne (biais) est
calculée pour chaque modèle :

.. code-block:: python

   from src.forecasting.calibration import estimate_oos_bias, apply_bias_correction

   biais = estimate_oos_bias(y_oos, y_pred)   # positif = surestimation
   corrigee = apply_bias_correction(prevision, biais)

Valeurs de biais OOS (MAD par once) :

.. list-table::
   :header-rows: 1
   :widths: 35 25 40

   * - Modèle
     - Biais (MAD)
     - Direction
   * - SARIMAX
     - −797
     - Sous-estimation
   * - Hybride ARIMA+XGB
     - −772
     - Sous-estimation (la plus faible)
   * - ARIMA
     - −865
     - Sous-estimation
   * - SARIMA
     - −927
     - Sous-estimation
   * - XGBoost
     - −1 438
     - Sous-estimation (la plus forte)
   * - LSTM
     - −1 573
     - Sous-estimation
   * - Prophet
     - −1 100
     - Sous-estimation

Tous les modèles sous-estiment systématiquement pendant la hausse de l'or 2024–2025.
La correction de biais décale la prévision de production vers le haut de la valeur
de biais estimée.

----

Intervalles probabilistes — ``probabilistic.py``
-------------------------------------------------

Les intervalles de confiance en éventail sont produits par **bootstrap sur les résidus** :

1. Échantillonnage de *n_samples = 500* trajectoires à partir des résidus ARIMA en
   échantillon.
2. Pour chaque trajectoire, ajout d'un bruit cumulatif mis à l'échelle par
   :math:`\sqrt{h / H}` à la prévision ponctuelle.
3. Les percentiles 2,5 et 97,5 forment l'intervalle de prévision à 95 %.

.. code-block:: python

   from src.forecasting.probabilistic import residual_bootstrap_intervals

   intervalle = residual_bootstrap_intervals(
       point_forecast=prevision,
       in_sample_residuals=arima.in_sample_residuals(),
       n_samples=500,
       alpha=0.05,
   )

L'intervalle s'élargit avec l'horizon, reflétant l'incertitude croissante des
prévisions récursives. Les résultats sont sauvegardés dans
``results/forecasts/forecast_intervals_2027_12.csv``.
