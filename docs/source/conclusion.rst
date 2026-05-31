.. _conclusion:

Conclusion et perspectives
==========================

Synthèse des résultats
-----------------------

Ce projet a construit un pipeline académique entièrement reproductible pour prévoir les
prix de l'or au Maroc à fréquence mensuelle. Les principales conclusions sont les
suivantes :

**1. SARIMAX est le meilleur modèle individuel**

Avec un RMSE moyen de **1 478 MAD**, un MAPE moyen de **6,49 %** et une précision
directionnelle de **60,9 %** sur 10 folds en fenêtre expansive, SARIMAX surpasse tous
les autres modèles pris séparément. Les facteurs clés de cette performance sont :

* L'inclusion directe du taux de change USD/MAD (coefficient +1 048) — le moteur
  dominant des variations de prix à court terme.
* Les régresseurs d'événements marocains qui capturent une saisonnalité prévisible
  de la demande, invisible aux modèles univariés.
* La procédure de sélection d'ordres ``auto_arima`` qui adapte les ordres ARIMA par fold.

**2. Ensemble_Weighted est recommandé en production**

La combinaison pondérée par l'inverse du RMSE de SARIMAX, XGBoost et Hybride
ARIMA+XGBoost réduit la sensibilité au mode d'échec de chaque modèle. Elle converge
en douceur vers ~51 000 MAD/once en décembre 2027 — une trajectoire économiquement
plausible grâce à l'amortissement de tendance.

**3. Les événements socio-culturels marocains ont un effet quantifiable**

Coefficients des événements issus du fit SARIMAX sur l'échantillon complet :

.. list-table::
   :header-rows: 1
   :widths: 35 20 45

   * - Événement
     - Coefficient (MAD)
     - Interprétation
   * - Ramadan
     - +22,0
     - Achats de bijoux avant l'Aïd
   * - Aïd Al-Adha
     - +70,7
     - Pic de demande de bijoux nuptiaux
   * - Saison des mariages
     - +48,1
     - Saison haute de 6 mois (avril–septembre)
   * - Aïd Al-Fitr
     - +28,1
     - Cadeaux de fin de Ramadan

Ces effets persistent après contrôle du change, de la macro et de la tendance,
confirmant que la saisonnalité culturelle est un moteur de prix indépendant et réel.

**4. Le deep learning (LSTM) et Prophet sous-performent sur ce jeu de données**

* Prophet atteint un MAPE de 12,3 % — presque le double de SARIMAX. Sa tendance
  linéaire par morceaux ne peut pas suivre les changements brusques de régime du
  prix de l'or.
* Le LSTM atteint un R² de 0,750 comme benchmark univarié, ce qui est respectable
  mais clairement en dessous des modèles statistiques avec variables exogènes.

**5. La méthodologie anti-fuite est critique**

Les trois suites de tests (``test_leakage_detection.py``, ``test_backtest_integrity.py``,
``test_xgboost_alignment.py``) démontrent que l'imputation globale naïve ou
l'entraînement sur un DataFrame combiné produit des résultats de backtest
matériellement différents (et optimistes). Chaque chiffre de performance dans ce
projet est exempt de fuite.

----

Limites
--------

**1. Sous-estimation en régime de forte volatilité**

Tous les modèles sous-estiment systématiquement les prix durant la hausse 2024–2026
(biais moyen de −772 à −1 573 MAD). Le régime récent est caractérisé par des
hausses mensuelles exceptionnellement rapides qui dépassent la plage de la
distribution historique d'entraînement.

**2. Hypothèses FX et macro pour les prévisions à long horizon**

Au-delà de 12 mois, le pipeline maintient les régresseurs FX et macro à leur
dernière valeur observée (``STRICT_EVALUATION_MODE = True``). De vraies prévisions
nécessiteraient des projections macro de BAM ou du FMI, qui comportent leur propre
incertitude.

**3. Le LSTM est univarié**

L'implémentation LSTM n'utilise que ``gold_price_mad`` comme entrée. Un LSTM
multivarié incorporant FX et régresseurs événementiels pourrait performer de façon
comparable à SARIMAX ; cette extension est identifiée comme travail futur.

**4. Disponibilité des données**

Le fichier ``macro_indicators.csv`` repose sur des constructions proxy lorsque les
flux primaires de BAM et FRED ne sont pas disponibles. Tout décalage de proxy
introduit du bruit dans les régresseurs macro.

**5. Sous-estimation des intervalles de prévision**

Les intervalles bootstrap sur les résidus sont très étroits (< 10 MAD à l'horizon
de 22 mois). Cela s'explique par leur échantillonnage depuis les résidus en
échantillon, qui ne reflètent pas les changements de régime ni l'incertitude
structurelle. Un intervalle prédictif approprié devrait incorporer l'incertitude
des paramètres et une analyse de scénarios.

----

Travaux futurs
---------------

Les extensions suivantes renforceraient le pipeline :

.. list-table::
   :header-rows: 1
   :widths: 32 68

   * - Extension
     - Description
   * - **LSTM multivarié**
     - Ajouter FX, événements et régresseurs macro à la séquence d'entrée du LSTM
   * - **Transformer / Temporal Fusion**
     - Explorer TFT (Lim et al., 2021) pour la prévision multi-pas à long horizon
       avec des poids d'attention interprétables
   * - **Prévisions par scénarios**
     - Exécuter le pipeline selon des scénarios de taux BAM (stable / hawkish /
       dovish) et des scénarios USD/MAD (trajectoires d'appréciation / dépréciation)
   * - **Intégration de données en temps réel**
     - Connexion à l'API BAM et aux flux World Gold Council pour des mises à jour
       mensuelles automatiques
   * - **Intervalles de prévision conformes**
     - Remplacer le bootstrap sur les résidus par la prédiction conforme pour des
       garanties de couverture sans hypothèse distributionnelle
   * - **Enquête sur les prix de détail marocains**
     - Ajouter une observation directe des prix de détail de l'or physique au Maroc
       pour valider l'identité ``gold_price_mad = gold_usd × usd_mad`` face aux
       frictions de marché (taxes à l'importation, marges des revendeurs)
   * - **Modélisation de la volatilité GARCH**
     - Ajuster un GARCH(1,1) sur les résidus SARIMAX pour améliorer le signal de
       régime de volatilité et produire des intervalles de prévision hétéroscédastiques

----

Références bibliographiques
-----------------------------

* Hyndman, R.J., & Athanasopoulos, G. (2021). *Forecasting: Principles and Practice*,
  3e édition. OTexts.
* Taylor, S.J., & Letham, B. (2018). Forecasting at scale.
  *The American Statistician*, 72(1), 37–45.
* Chen, T., & Guestrin, C. (2016). XGBoost: A scalable tree boosting system.
  *ACM KDD*, 785–794.
* Lim, B., Arık, S.Ö., Loeff, N., & Pfister, T. (2021). Temporal fusion transformers
  for interpretable multi-horizon time series forecasting.
  *International Journal of Forecasting*, 37(4), 1748–1764.
* Box, G.E.P., Jenkins, G.M., Reinsel, G.C., & Ljung, G.M. (2015).
  *Time Series Analysis: Forecasting and Control*, 5e édition. Wiley.
* Bank Al-Maghrib (2024). *Rapport annuel sur la situation économique, monétaire et
  financière*. BAM.

----

Liste de contrôle pour la reproductibilité
--------------------------------------------

Avant de relancer le pipeline :

.. code-block:: bash

   # 1. Vérifier l'environnement
   pip install -r requirements.txt

   # 2. Lancer les tests unitaires
   pytest tests/ -v

   # 3. Lancer le pipeline complet
   python main.py

   # 4. Vérifier les sorties
   ls results/tables/model_ranking_backtest.csv
   ls results/forecasts/future_forecasts_to_2027_12.csv
   ls results/models/pipeline_run_metadata.json

Le fichier ``pipeline_run_metadata.json`` enregistre les versions exactes des packages,
le meilleur modèle, les listes de features et les paramètres de prétraitement par fold
utilisés lors de l'exécution.
