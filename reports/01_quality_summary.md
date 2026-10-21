# Résumé de qualité des données - Rossmann

## Données chargées
- `train.csv`: 1 017 209 lignes, 9 colonnes
- `store.csv`: 1 115 lignes, 10 colonnes
- Période observée: 2013-01-01 à 2015-07-31
- 1 115 magasins couverts, sans magasin manquant dans la jointure

## Anomalies détectées

| Problème | Observation | Décision |
|---|---|---|
| Jours fermés | 172 817 lignes avec `Open = 0` (16,99 %) | Exclure de l’entraînement et prévoir 0 sur ces jours |
| Ventes nulles avec magasin ouvert | 54 lignes avec `Open = 1` et `Sales = 0` | Vérifier, conserver seulement si contexte métier validé |
| `StateHoliday` | Mélange de types, valeurs normalisées `0`, `a`, `b`, `c` | Standardiser avant modélisation |
| `CompetitionDistance` | 3 valeurs manquantes | Imputer par la médiane + indicateur de manquant |
| `CompetitionOpenSince*` | 354 valeurs manquantes | Encodage spécifique "non applicable" |
| `Promo2Since*` | 544 valeurs manquantes | Encodage spécifique "non applicable" |
| Doublons | Aucun doublon détecté | Pas de correction nécessaire |
| Couverture temporelle | 0 date manquante, 0 magasin incomplet | Les lags par magasin peuvent être calculés sans trou |

## Règles métier retenues
1. Les jours fermés sont traités séparément et ne participent pas à l’entraînement.
2. Les ventes des jours fermés sont fixées à 0 dans la prédiction.
3. Les variables de concurrence ou de promo avec absence logique sont encodées comme non applicables, pas imputées aveuglément.
4. Les données temporelles sont cohérentes sans trous de calendrier sur la période étudiée.
