.. _preprocessing:

Prétraitement
=============

La couche de prétraitement est responsable de :

1. Charger et fusionner les quatre fichiers CSV bruts.
2. Imposer un ``DatetimeIndex`` mensuel propre.
3. Appliquer une **imputation par fold** — sans statistiques globales, sans fuite de données futures.

Tous les modules se trouvent dans ``src/preprocessing/``.

----

Chargement des données — ``data_loader.py``
--------------------------------------------

Points d'entrée
~~~~~~~~~~~~~~~

.. code-block:: python

   from src.preprocessing.data_loader import load_raw_data, load_and_prepare_data

   df = load_raw_data()          # recommandé — retourne le DataFrame fusionné brut avec NaN intacts
   df = load_and_prepare_data()  # alias rétrocompatible de load_raw_data()

``load_raw_data()`` effectue les étapes suivantes dans l'ordre :

1. **Chargement des prix de l'or** (``_load_gold``) — lit ``data/gold_prices.csv``,
   renomme les colonnes, convertit en numérique, impose l'index mensuel.
2. **Chargement USD/MAD** (``_load_usd_mad``) — applique le parsing décimal européen,
   impose l'index mensuel.
3. **Chargement des événements** (``_load_events``) — lit les colonnes d'intensité ;
   convertit en numérique.
4. **Jointure interne** or × FX × événements sur l'index mensuel.
5. **Calcul de la cible** : ``df["gold_price_mad"] = df["gold_price_usd"] * df["usd_mad"]``.
6. **Jointure gauche macro** (``_load_macro_optional``) si ``data/macro_indicators.csv``
   existe ; les colonnes macro manquantes sont silencieusement ignorées.

.. warning::

   ``load_raw_data()`` laisse intentionnellement les valeurs ``NaN`` en place.
   N'appliquez **pas** ``fillna`` ou ``interpolate`` sur le DataFrame fusionné.
   Toute imputation doit se faire à l'intérieur de ``FoldPreprocessor.fit()`` /
   ``transform()`` pour éviter le biais de look-ahead.

Nettoyage — ``cleaners.py``
-----------------------------

.. code-block:: python

   from src.preprocessing.cleaners import (
       parse_european_decimal,
       to_monthly_datetime,
       enforce_monthly_index,
   )

``parse_european_decimal(valeur)``
   Convertit une chaîne comme ``"9,85"`` ou ``"  9.85 "`` en ``float`` Python.
   Gère les espaces, virgules, caractères non numériques et les entrées ``NaN``.

``to_monthly_datetime(series)``
   Convertit une ``pd.Series`` de chaînes de dates en ``pd.DatetimeIndex`` normalisé
   au début du mois (fréquence ``MS``) via ``pd.Period("M").to_timestamp()``.

``enforce_monthly_index(df, date_col)``
   * Parse la colonne de date.
   * Trie par date et supprime les mois en double (garde la dernière observation).
   * Réindexe sur une plage mensuelle complète afin que les mois manquants
     apparaissent comme ``NaN`` plutôt que d'être silencieusement absents.

----

Imputation par fold — ``fold_preprocessor.py``
-----------------------------------------------

Le ``FoldPreprocessor`` est le composant anti-fuite le plus important du pipeline.
C'est un transformateur à état, de style scikit-learn, **ajusté sur le fold d'entraînement
uniquement** puis appliqué séparément sur l'entraînement et le test.

.. code-block:: python

   from src.preprocessing.fold_preprocessor import FoldPreprocessor, prepare_fold_frames

Règles de conception
~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 20 80

   * - Contexte
     - Règle
   * - **Moitié entraînement**
     - Remplissage par propagation vers l'avant dans la fenêtre d'entraînement, puis
       remplissage des ``NaN`` restants avec la **médiane** de ce fold (les colonnes
       événementielles utilisent la médiane de la série propagée ; les colonnes macro
       utilisent la médiane numérique).
   * - **Moitié test**
     - La *première* valeur manquante est remplie avec la **dernière valeur observée
       en entraînement** (``last_train_value_``), puis propagation vers l'avant dans
       la fenêtre de test. Les ``NaN`` restants sont remplis avec la valeur de
       remplissage du fold d'entraînement.
       **Le remplissage en arrière depuis le futur n'est jamais autorisé.**
   * - **Événements**
     - Médiane de la série d'entraînement propagée (pas la moyenne, pour être robuste
       aux zéros).
   * - **Macro**
     - Médiane numérique des observations d'entraînement.

API
~~~

.. code-block:: python

   prep = FoldPreprocessor(fold_id=0)
   prep.fit(train_raw)                               # apprend fill_values_, last_train_value_
   train_clean = prep.transform(train_raw, split_role="train")
   test_clean  = prep.transform(test_raw,  split_role="test")

   # Sérialisation pour la traçabilité
   meta = prep.to_dict()   # stocké dans backtest_fold_metadata.json

``prepare_fold_frames(train_raw, test_raw, fold_id)``
   Encapsuleur pratique qui :

   1. Instancie et ajuste un ``FoldPreprocessor``.
   2. Transforme les deux moitiés.
   3. Les concatène et exécute ``build_features()`` afin que les colonnes de lag au
      début de la fenêtre de test puissent regarder en arrière dans l'historique
      d'entraînement.
   4. Redécupe le DataFrame enrichi en DataFrames propres d'entraînement et de test.

   Retourne ``(train_df, test_df, prep)``.

Tests anti-fuite
~~~~~~~~~~~~~~~~

Trois tests unitaires valident les garanties d'imputation :

.. code-block:: bash

   pytest tests/test_leakage_detection.py -v
   pytest tests/test_backtest_integrity.py -v

Assertions principales :

* ``test_test_imputation_does_not_use_future_bfill`` — les valeurs ``NaN`` de l'ensemble
  de test doivent être remplies avec la dernière valeur d'entraînement, pas une
  observation future de test.
* ``test_global_bfill_would_differ_from_fold_safe`` — démontre que le ``bfill`` global
  naïf produit des résultats différents (avec fuite).
* ``test_imputer_uses_train_stats_only`` — vérifie que ``fill_values_`` est calculé
  uniquement à partir des données d'entraînement.
* ``test_no_target_leakage_in_features`` — la colonne cible et les colonnes brutes
  ``gold_price_usd`` / ``usd_mad`` n'apparaissent pas dans l'ensemble de features ML.

----

Structure des folds — génération des découpages
-------------------------------------------------

Les découpages sont générés par ``generate_backtest_splits()`` dans ``backtesting.py`` :

.. list-table::
   :header-rows: 1
   :widths: 35 65

   * - Paramètre
     - Valeur (défaut)
   * - ``initial_train``
     - 180 mois (15 ans)
   * - ``test_horizon``
     - 12 mois par fold
   * - ``step``
     - 12 mois (avancée annuelle)
   * - ``strategy``
     - ``"expanding"`` — la fenêtre d'entraînement s'agrandit à chaque fold

Avec 307 observations et un entraînement initial de 180, **10 folds** sont produits :

.. list-table::
   :header-rows: 1
   :widths: 8 22 22 28 14 6

   * - Fold
     - Début entraînement
     - Fin entraînement
     - Période de test
     - N entraînement
     - N test
   * - 0
     - 2000-08
     - 2015-07
     - 2015-08 → 2016-07
     - 180
     - 12
   * - 1
     - 2000-08
     - 2016-07
     - 2016-08 → 2017-07
     - 192
     - 12
   * - 2
     - 2000-08
     - 2017-07
     - 2017-08 → 2018-07
     - 204
     - 12
   * - 3
     - 2000-08
     - 2018-07
     - 2018-08 → 2019-07
     - 216
     - 12
   * - 4
     - 2000-08
     - 2019-07
     - 2019-08 → 2020-07
     - 228
     - 12
   * - 5
     - 2000-08
     - 2020-07
     - 2020-08 → 2021-07
     - 240
     - 12
   * - 6
     - 2000-08
     - 2021-07
     - 2021-08 → 2022-07
     - 252
     - 12
   * - 7
     - 2000-08
     - 2022-07
     - 2022-08 → 2023-07
     - 264
     - 12
   * - 8
     - 2000-08
     - 2023-07
     - 2023-08 → 2024-07
     - 276
     - 12
   * - 9
     - 2000-08
     - 2024-07
     - 2024-08 → 2025-07
     - 288
     - 12

Les métadonnées de chaque fold (dates d'entraînement/test, valeurs de remplissage,
dernières valeurs d'entraînement) sont sauvegardées dans
``results/models/backtest_fold_metadata.json`` pour une reproductibilité totale.
