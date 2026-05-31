.. _metrics:

Référence des métriques
========================

Toutes les métriques d'évaluation sont implémentées dans ``src/evaluation/metrics.py``.

Alignement des séries
----------------------

Avant tout calcul de métrique, ``_align(y_true, y_pred)`` intersecte les deux séries
sur leur index, convertit en ``float64``, et supprime toute ligne où l'une ou l'autre
valeur est ``NaN``. Cela prévient les erreurs silencieuses liées aux désalignements
d'index entre les sorties des folds.

----

Métriques principales
----------------------

RMSE — Racine de l'Erreur Quadratique Moyenne
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. math::

   \text{RMSE} = \sqrt{\frac{1}{n} \sum_{t=1}^{n} (y_t - \hat{y}_t)^2}

Pénalise les grandes erreurs de façon quadratique. C'est la **métrique principale de
classement** pour la sélection des modèles car elle est sensible aux rares grandes
erreurs qui comptent le plus en pratique. Exprimée en MAD par once troy.

MAE — Erreur Absolue Moyenne
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. math::

   \text{MAE} = \frac{1}{n} \sum_{t=1}^{n} |y_t - \hat{y}_t|

Plus robuste aux valeurs aberrantes que le RMSE. Exprimée en MAD.

MAPE — Erreur Absolue Moyenne en Pourcentage
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. math::

   \text{MAPE} = \frac{100}{n} \sum_{t=1}^{n} \left| \frac{y_t - \hat{y}_t}{y_t} \right|

Adimensionnelle ; utile pour comparer sur des périodes avec des niveaux de prix
différents. Le meilleur modèle (SARIMAX) atteint **6,49 %** de MAPE moyen, ce qui
signifie que les erreurs mensuelles typiques représentent environ 6,5 % du prix réel.

R² — Coefficient de détermination
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. math::

   R^2 = 1 - \frac{\sum (y_t - \hat{y}_t)^2}{\sum (y_t - \bar{y})^2}

.. warning::

   Pour l'évaluation par backtest glissant, le R² est calculé par rapport à la
   **moyenne hors échantillon** (la moyenne des valeurs actuelles du fold de test),
   pas la moyenne globale. Comme le prix de l'or est fortement croissant, la moyenne
   OOS est toujours très éloignée des valeurs réelles, ce qui donne un **R² négatif**
   pour tous les modèles même lorsque les prévisions sont précises.

   Le **R² bootstrap** (calculé sur les prévisions OOS regroupées) est la mesure la
   plus pertinente : SARIMAX atteint 0,859 [0,827 – 0,892].

Biais
~~~~~~

.. math::

   \text{Biais} = \frac{1}{n} \sum_{t=1}^{n} (\hat{y}_t - y_t)

Erreur de prévision moyenne signée. Positif = surestimation systématique ;
négatif = sous-estimation. Tous les modèles présentent un biais négatif (−772 à
−1 573 MAD), reflétant la difficulté à capturer le momentum de la hausse 2024–2025.

Précision directionnelle
~~~~~~~~~~~~~~~~~~~~~~~~~

.. math::

   \text{PD} = \frac{100}{n-1} \sum_{t=2}^{n}
   \mathbf{1}\!\left[\text{signe}(y_t - y_{t-1}) = \text{signe}(\hat{y}_t - \hat{y}_{t-1})\right]

Pourcentage de mois où le modèle prédit correctement le sens du mouvement de prix
(hausse ou baisse). Une prévision aléatoire atteint ~50 %.

SARIMAX et XGBoost atteignent tous deux **60,9 %** de précision directionnelle —
environ 11 points de pourcentage au-dessus de l'aléatoire.

----

Intervalles de confiance bootstrap
------------------------------------

Implémentés dans ``src/evaluation/bootstrap.py`` avec 500 itérations de rééchantillonnage
et ``np.random.default_rng(42)`` pour la reproductibilité.

.. code-block:: python

   from src.evaluation.bootstrap import bootstrap_metric_ci, bootstrap_all_models

   ic = bootstrap_metric_ci(y_true, y_pred, metric="rmse", n_samples=500)
   # Retourne {"point": ..., "low": ..., "high": ...}

   df = bootstrap_all_models(y_true, dict_previsions)
   # Retourne un DataFrame en format long : model, metric, point, ci_low, ci_high

----

Références naïves
------------------

Trois références naïves sont incluses dans chaque fold du backtest pour contextualiser
les performances des modèles (``src/evaluation/baselines.py``) :

.. list-table::
   :header-rows: 1
   :widths: 28 72

   * - Référence
     - Description
   * - ``Référence_DernièreValeur``
     - Répète la dernière valeur d'entraînement observée pour tous les *h* pas.
       Atteint 0 % de précision directionnelle par construction.
   * - ``Référence_MoyenneEntraînement``
     - Utilise la moyenne du fold d'entraînement pour tous les *h* pas.
       Pire modèle global (MAPE 44,8 %).
   * - ``Référence_Drift``
     - Projette une tendance linéaire constante du premier au dernier point
       d'entraînement. Compétitif avec SARIMAX sur le RMSE (1 487) et la précision
       directionnelle (60,9 %) — soulignant à quel point la prévisibilité de l'or
       provient de sa forte tendance.

----

Métriques regroupées vs moyennées par fold
-------------------------------------------

Deux modes d'agrégation sont supportés :

**Moyennées par fold** (``model_ranking_backtest.csv``) :
   Calcule la métrique indépendamment pour la fenêtre de 12 mois de chaque fold, puis
   prend la moyenne arithmétique et l'écart-type sur les folds. Robuste à
   l'hétéroscédasticité au niveau des folds ; c'est la **métrique de classement
   principale**.

**OOS regroupé** (``model_ranking_pooled_oos.csv``) :
   Concatène toutes les 10 × 12 = 120 observations de test en une seule série et
   calcule la métrique une fois. Plus sensible aux folds récents à forte volatilité
   (qui dominent les résidus regroupés) mais fournit un résumé de performance en
   un seul chiffre.
