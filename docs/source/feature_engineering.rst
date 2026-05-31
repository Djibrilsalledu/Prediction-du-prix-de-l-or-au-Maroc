.. _feature_engineering:

Ingénierie des features
========================

Tout le code d'ingénierie des features se trouve dans ``src/feature_engineering/``.
Les deux modules clés sont ``features.py`` (features en échantillon) et
``exogenous_future.py`` (construction des variables exogènes hors échantillon).

La règle cardinale est **zéro look-ahead** : chaque feature calculée à l'instant *t*
n'utilise que les informations disponibles jusqu'à *t − 1* inclus.

----

Features en échantillon — ``features.py``
------------------------------------------

Point d'entrée
~~~~~~~~~~~~~~

.. code-block:: python

   from src.feature_engineering.features import (
       build_features,
       get_ml_feature_columns,
       training_row_mask,
   )

   df_enrichi  = build_features(df)                          # ajoute ~85 colonnes
   feat_cols   = get_ml_feature_columns(df_enrichi)          # liste des colonnes ML-sûres
   masque      = training_row_mask(df_enrichi, feat_cols)    # filtre de chauffe

``build_features(df)``
~~~~~~~~~~~~~~~~~~~~~~

Construit la matrice complète de features à partir d'un ``DataFrame`` prétraité.
Toutes les opérations décalent la série source d'au moins 1 pas avant de calculer
les colonnes dérivées.

**Lags de la cible**

.. code-block:: text

   gold_price_mad_lag_1, _lag_3, _lag_6, _lag_12

**Statistiques glissantes** (appliquées à ``gold_price_mad.shift(1)``) :

.. code-block:: text

   gold_price_mad_roll_mean_3,  _roll_std_3
   gold_price_mad_roll_mean_6,  _roll_std_6
   gold_price_mad_roll_mean_12, _roll_std_12

**Momentum et rendements** :

.. list-table::
   :header-rows: 1
   :widths: 45 55

   * - Feature
     - Formule
   * - ``gold_price_mad_pct_change_1``
     - ``y_{t-1} / y_{t-2} - 1``
   * - ``gold_price_mad_pct_change_12``
     - ``y_{t-1} / y_{t-13} - 1``
   * - ``gold_price_mad_momentum_3``
     - ``y_{t-1} - y_{t-4}``
   * - ``gold_price_mad_momentum_12``
     - ``y_{t-1} - y_{t-13}``
   * - ``gold_price_mad_roll_vol_6``
     - Écart-type glissant des rendements mensuels, fenêtre = 6
   * - ``gold_price_mad_roll_vol_12``
     - Écart-type glissant des rendements mensuels, fenêtre = 12
   * - ``gold_price_mad_trend_strength_12``
     - ``|y_{t-1} - y_{t-13}| / y_{t-13}``
   * - ``gold_price_mad_roll_return_3``
     - Rendement en pourcentage sur 3 mois décalé

**Encodage cyclique du temps** (mois de l'année, sans biais ordinal) :

.. code-block:: text

   month_sin = sin(2π × mois / 12)
   month_cos = cos(2π × mois / 12)
   quarter, year

**Features événementielles marocaines** (par événement : ``ramadan``, ``eid_alfitr``,
``eid_aladha``, ``wedding_season``, ``mre_season``) :

.. code-block:: text

   <événement>_lag_1, _lag_3
   <événement>_roll_mean_3, _roll_mean_12
   <événement>_cumulative_12

**Lags et interactions FX / or USD** :

.. code-block:: text

   gold_price_usd_lag_1, gold_price_usd_lag1
   usd_mad_lag_1, usd_mad_lag1
   gold_price_usd_lag1_x_usd_mad_lag1        ← terme d'interaction
   usd_mad_pct_change_1
   usd_mad_roll_mean_3 / _6 / _12
   usd_mad_roll_std_3  / _6 / _12

**Termes d'interaction macro** (pour chaque variable macro ``m``) :

.. code-block:: text

   <m>_pct_change_1
   <m>_roll_mean_6
   <m>_roll_vol_6
   gold_price_usd_lag1_x_<m>_lag1

Au total, cela produit **~85 features** (le nombre exact varie selon la disponibilité
des colonnes macro).

``get_ml_feature_columns(df, frozen=None)``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Retourne la liste des colonnes sûres pour les modèles ML en excluant :

* ``gold_price_mad`` (cible — constituerait une fuite)
* ``gold_price_usd``, ``usd_mad`` (prédicteurs bruts contemporains — fuite)
* Toute colonne non numérique ou ``Unnamed``

Accepte optionnellement une liste ``frozen`` (l'ensemble de features du fit final sur
l'échantillon complet) pour garantir un ordre de colonnes identique à l'inférence.

``training_row_mask(feature_df, feat_cols)``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Retourne une ``Series`` booléenne qui vaut ``True`` uniquement pour les lignes où :

1. L'index de ligne dépasse les premiers ``MAX_LAG = 12`` enregistrements (période de
   chauffe).
2. Toutes les colonnes de features sont non-``NaN``.

Cela empêche les modèles de gradient boosting de s'entraîner sur des fenêtres de lag
partiellement remplies.

----

Construction des variables exogènes futures — ``exogenous_future.py``
----------------------------------------------------------------------

Lors de la génération de prévisions au-delà de la dernière date observée, les colonnes
exogènes doivent être projetées vers l'avenir. Le module
``src/feature_engineering/exogenous_future.py`` propose deux stratégies :

Point d'entrée
~~~~~~~~~~~~~~

.. code-block:: python

   from src.feature_engineering.exogenous_future import build_future_exogenous

   exog_futur = build_future_exogenous(
       historique,
       last_observed=df.index.max(),
       forecast_end="2027-12-01",
       exog_cols=["usd_mad", "ramadan", ...],
       allow_extrapolation=False,   # STRICT_EVALUATION_MODE
   )

Projection des événements (``extend_events``)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Les colonnes d'événements marocains sont **déterminées par le calendrier** : leurs
valeurs futures peuvent être approchées à partir du calendrier islamique. La formule
de mélange est :

.. math::

   \hat{e}_{t} = 0{,}7 \times \bar{e}_{\text{mois}} + 0{,}3 \times e_{\text{an dernier, mois}}

où :math:`\bar{e}_{\text{mois}}` est l'intensité historique moyenne pour ce mois
calendaire. Le résultat est écrêté à [0, 1].

Projection FX et macro (``extend_numeric_exog``)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Pour ``usd_mad``, ``oil_brent_usd``, ``dxy_index``, etc., une **tendance linéaire
amortie** est utilisée :

.. math::

   \hat{x}_{t+h} = \phi^h \cdot \hat{x}^{\text{linéaire}}_{t+h} + (1 - \phi^h) \cdot x_t

avec :math:`\phi = 0{,}95` et un plafond de ±12 % autour de la dernière valeur
observée.

.. warning::

   L'extrapolation FX/macro est **désactivée par défaut** lorsque
   ``STRICT_EVALUATION_MODE = True`` (``src/utils/config.py``).
   En mode strict, les colonnes FX et macro sont maintenues à leur dernière valeur
   d'entraînement observée sur tout l'horizon de prévision. Cela évite des hypothèses
   non vérifiées sur les conditions macroéconomiques futures lors du backtesting.

   Définissez ``ALLOW_EXOG_EXTRAPOLATION = True`` (ou passez
   ``allow_extrapolation=True``) uniquement pour la prévision finale de production.

Configuration
~~~~~~~~~~~~~

Tous les paramètres d'ingénierie des features sont centralisés dans
``src/utils/config.py`` :

.. list-table::
   :header-rows: 1
   :widths: 38 15 47

   * - Constante
     - Valeur par défaut
     - Description
   * - ``MAX_LAG``
     - ``12``
     - Nombre minimal de lignes de chauffe pour le masque d'entraînement
   * - ``STRICT_EVALUATION_MODE``
     - ``True``
     - Désactive l'extrapolation FX/macro lors du backtesting
   * - ``ALLOW_EXOG_EXTRAPOLATION``
     - ``False``
     - Dérivé de ``not STRICT_EVALUATION_MODE``
   * - ``EVENT_COLUMNS``
     - (liste)
     - Noms des cinq colonnes d'événements marocains
   * - ``MACRO_COLUMNS``
     - (liste)
     - Noms des cinq colonnes d'indicateurs macro
