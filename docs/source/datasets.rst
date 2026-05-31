.. _datasets:

Jeux de données
===============

Les quatre jeux de données partagent une **fréquence mensuelle** avec des horodatages
normalisés au début du mois (``MS``). L'index commun couvre **août 2000 à février 2026**
(307 observations). Aucune imputation globale n'est appliquée au chargement — les valeurs
manquantes restent ``NaN`` jusqu'au prétraitement par fold.

.. note::

   Le point d'entrée pour le chargement des données est ``src/preprocessing/data_loader.py``.
   Appelez ``load_raw_data()`` pour obtenir le ``DataFrame`` fusionné avec la colonne cible.

----

gold_prices.csv
---------------

**Cours spot international de l'or en USD par once troy.**

.. list-table::
   :header-rows: 1
   :widths: 20 15 65

   * - Colonne
     - Type
     - Description
   * - ``date``
     - ``datetime``
     - Horodatage début de mois (ex. ``2000-08-01``)
   * - ``gold_price_usd``
     - ``float``
     - Prix de l'or en USD par once troy (XAU/USD)

**Source :** World Gold Council / Yahoo Finance (``GC=F``).

**Caractéristiques principales :**

* Tendance haussière forte : de ~270 $ (août 2000) à ~2 900 $ (février 2026).
* 57 points de rupture structurels détectés par le détecteur de régimes z-score glissant.
* Non stationnaire : statistique ADF = +2,30, p-valeur = 0,999 → différenciation nécessaire (d = 1).

----

usd_mad.csv
-----------

**Taux de change USD / Dirham marocain.**

.. list-table::
   :header-rows: 1
   :widths: 20 15 65

   * - Colonne
     - Type
     - Description
   * - ``date``
     - ``datetime``
     - Horodatage début de mois
   * - ``usd_mad``
     - ``float``
     - Taux de change USD/MAD mensuel moyen (nombre de MAD pour 1 USD)

**Source :** Bank Al-Maghrib (BAM) / Banque mondiale.

**Note de parsing :** Le fichier brut peut utiliser la notation décimale européenne
(séparateur virgule). Le chargeur applique ``parse_european_decimal()`` depuis
``src/preprocessing/cleaners.py``.

**Rôle dans le pipeline :**

La variable cible est définie comme :

.. math::

   \text{gold\_price\_mad} = \text{gold\_price\_usd} \times \text{usd\_mad}

``usd_mad`` est également utilisé directement comme régresseur exogène dans SARIMAX,
avec un coefficient estimé de **+1 048** — le plus grand effet exogène du modèle.

----

moroccan_events.csv
-------------------

**Intensité mensuelle des événements socio-culturels marocains (échelle 0 – 1).**

.. list-table::
   :header-rows: 1
   :widths: 22 12 66

   * - Colonne
     - Plage
     - Description
   * - ``date``
     - —
     - Horodatage début de mois
   * - ``ramadan``
     - [0, 1]
     - Chevauchement fractionnaire du mois lunaire du Ramadan avec le mois calendaire.
       Les achats de bijoux culminent avant l'Aïd Al-Fitr ; coefficient SARIMAX = **+22,0**.
   * - ``eid_alfitr``
     - [0, 1]
     - Intensité de l'Aïd Al-Fitr. La concentration des cadeaux génère des pics courts
       de demande d'or ; coefficient = **+28,1**.
   * - ``eid_aladha``
     - [0, 1]
     - Intensité de l'Aïd Al-Adha. Les dépenses se dirigent vers le bétail ;
       impact moindre sur l'or ; coefficient = **+70,7**.
   * - ``wedding_season``
     - [0, 1]
     - Saison des mariages marocains (avril–septembre). Les bijoux en or constituent
       une part importante de la dot (*mahr*) ; coefficient = **+48,1**.
   * - ``mre_season``
     - [0, 1]
     - Saison de retour des Marocains Résidant à l'Étranger (juin–août). Les transferts
       d'argent convertis en MAD stimulent la demande locale d'or.

**Source :** Calcul du calendrier islamique + enquêtes de marché locales.

**Utilisation dans les prévisions futures :**

Les colonnes événementielles sont projetées vers l'avenir selon une approche de
*profil saisonnier* (``src/feature_engineering/exogenous_future.py``) :

* Calcul de l'intensité historique moyenne par mois calendaire (moyenne par mois de l'année).
* Mélange à 70 % de la moyenne long terme et 30 % du schéma observé l'année précédente.
* Écrêtage à [0, 1].

----

macro_indicators.csv
--------------------

**Régresseurs macroéconomiques optionnels — construits à la demande.**

.. list-table::
   :header-rows: 1
   :widths: 25 12 63

   * - Colonne
     - Unité
     - Description
   * - ``oil_brent_usd``
     - USD/baril
     - Moyenne mensuelle du pétrole Brent. Co-mouvement positif avec l'or en tant que
       matière première. Coefficient SARIMAX = **−17,7** (effet substitution / risk-on).
   * - ``dxy_index``
     - Indice
     - Indice du dollar américain (DXY). Forte corrélation négative avec le prix de
       l'or en USD (−0,75 dans la matrice de corrélation) ; coefficient = **−79,2**.
   * - ``fed_funds_rate``
     - %
     - Taux des fonds fédéraux effectif. Des taux US plus élevés augmentent le coût
       d'opportunité de détenir de l'or ; coefficient = **−0,29**.
   * - ``inflation_morocco``
     - % en glissement annuel
     - Inflation IPC au Maroc. Utilisée comme indicateur de pression côté demande.
   * - ``policy_rate_bam``
     - %
     - Taux directeur de Bank Al-Maghrib. Réponse endogène à l'inflation ; corrélé à
       ``inflation_morocco`` (voir matrice de corrélation).

**Construction du fichier macro :**

.. code-block:: bash

   python main.py --build-macro

Cette commande exécute ``src/preprocessing/build_macro.py``, qui récupère le Brent et
le DXY depuis ``yfinance`` et construit des séries proxy pour le taux Fed, l'inflation
marocaine et le taux BAM. Si ``yfinance`` n'est pas disponible, toutes les colonnes macro
sont remplies avec ``NaN`` et le pipeline utilise uniquement les régresseurs événementiels.

**Traitement des valeurs manquantes :**

Les colonnes macro ne sont **pas** imputées globalement. Les valeurs manquantes aux
frontières des folds sont gérées par ``FoldPreprocessor``
(voir :ref:`preprocessing`) :

* *Moitié entraînement* : remplissage par propagation vers l'avant dans le fold,
  puis remplissage des ``NaN`` restants avec la médiane calculée sur le fold d'entraînement.
* *Moitié test* : report de la dernière valeur observée en entraînement ; jamais de
  remplissage en arrière depuis les observations futures.

----

Colonne cible dérivée
-----------------------

Après fusion des quatre sources, le chargeur calcule :

.. code-block:: python

   df["gold_price_mad"] = df["gold_price_usd"] * df["usd_mad"]

Cette colonne est le **seul objectif de prévision**. Toutes les features et sorties de
modèles sont exprimées en MAD par once troy.

Statistiques descriptives (échantillon complet)
------------------------------------------------

.. list-table::
   :header-rows: 1
   :widths: 30 18 18 18 16

   * - Série
     - Min
     - Moyenne
     - Max
     - Obs.
   * - ``gold_price_usd``
     - ~268
     - ~1 100
     - ~2 940
     - 307
   * - ``usd_mad``
     - ~8,5
     - ~9,6
     - ~10,8
     - 307
   * - ``gold_price_mad``
     - ~2 300
     - ~10 800
     - ~43 200
     - 307
   * - ``oil_brent_usd``
     - ~18
     - ~76
     - ~130
     - 307
   * - ``dxy_index``
     - ~72
     - ~91
     - ~115
     - 307
