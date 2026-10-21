# Modélisation Rossmann - Résumé final

## Résultat validé

Le modèle LightGBM a été évalué sur une fenêtre temporelle de 6 semaines, en conservant le même protocole de validation que le baseline.

### Scores

- MAE : 372.15
- RMSE : 549.73
- RMSPE : 0.0742
- temps d'entraînement : 20.29 s

### Baseline de référence

- MAE : 1240.35
- RMSE : 1649.65
- RMSPE : 0.2332

### Amélioration

- MAE : -69.99%
- RMSE : -66.70%
- RMSPE : -68.15%

## Conclusion

Le modèle LightGBM est nettement supérieur au baseline temporel et fournit des prédictions bien plus précises pour les ventes des magasins Rossmann. Il capte avec succès les variations liées aux promos, au type de magasin, aux périodes de concurrence et aux patterns hebdomadaires/saisonniers.

Ce projet démontre correctement le pipeline complet d'un projet de forecasting : nettoyage, validation, ingénierie des features, benchmark et modélisation. Le modèle final est prêt pour une présentation dans un portfolio data science, avec des résultats clairs, reproductibles et supérieurs au benchmark de référence.

## Fichier visuel

- Graphique de comparaison : reports/model_vs_baseline_comparison.png
