# Détection d'illustrations — évaluation et modèles

Objectif : choisir le meilleur détecteur d'illustrations sur vues Gallica numérisées.
On détecte les **illustrations** : zéro, une ou plusieurs par vue. Les boîtes produites
alimentent ensuite l'étape de classification, qui ne relève pas de ce document.

---

## 1. Métriques d'évaluation

C'est la partie qui décide. Tant que ces métriques ne sont pas implémentées et figées,
lancer un entraînement ne sert à rien : les résultats ne seraient pas comparables.

### L'IoU, socle de toutes les métriques

L'**IoU** (intersection sur union) entre une boîte prédite et une boîte de vérité terrain est
le **critère d'appariement** de toute la chaîne. Elle n'apparaît pas comme une ligne séparée
parce qu'elle est _dans_ chaque métrique ci-dessous :

- **AP50** = average precision en comptant correcte toute prédiction dont l'**IoU ≥ 0.50**
- **mAP50-95** = la même, moyennée sur dix seuils d'IoU de 0.50 à 0.95

L'enchaînement complet : **IoU → appariement → comptage TP/FP/FN → précision/rappel → AP**.

### Métrique principale

**Précision à rappel fixé, classe `Illustration`** — typiquement P@R=0.90.

Le problème métier est l'**hallucination d'illustrations** : le dataset livre 372 vues où
le modèle précédent a inventé une illustration. On veut donc savoir _combien de fausses
détections coûte un niveau de rappel donné_. C'est cette valeur qui classe les modèles.

### Métriques standard

| Métrique                                 | Pourquoi                                                           |
| ---------------------------------------- | ------------------------------------------------------------------ |
| **mAP50-95**                             | Comparabilité avec la littérature                                  |
| **AP50 par classe**                      | Sépare `Illustration` de `Texte` ; la moyenne des deux masque tout |
| **Précision / rappel / F1** à seuil fixé | Le point de fonctionnement réel en production                      |

### Stratifications — obligatoires

Une métrique globale sur ce dataset est trompeuse. Deux découpages à produire systématiquement :

**Par taille de boîte.** La distribution des aires est fortement bimodale — p25 = 3 % de la
page, p75 = 74 %. Environ 3 700 boîtes couvrent plus de la moitié de la vue, ~1 100 en
couvrent moins de 1 %. Les grosses boîtes sont faciles et écrasent la moyenne : **sans
stratification, le régime « petites illustrations » est invisible.** Utiliser les quantiles
d'aire comme bornes de strates.

**Par tag de contenu** (`photographie`, `comic_book`, `film_roll`, `plan`, `poster`…).
Indique quel modèle échoue sur quelle sous-population — c'est le matériau de l'analyse
d'erreurs.

### Métriques spécifiques au problème

| Métrique                                                                  | Ce qu'elle capture                                                                                                                                                                                   |
| ------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Taux de faux positifs sur vues sans illustration**                      | 398 vues n'en contiennent aucune (dont 157 taguées `empty` : pages de garde, papiers marbrés). Mesure directe et lisible de l'hallucination, indépendante de tout seuil de mAP                       |
| **IoU moyen des boîtes appariées** — à lire _avec_ le rappel, jamais seul | Les boîtes détectées alimentent l'étape de **classification** en aval. Une boîte trop lâche produit un recadrage inexploitable : la qualité de localisation compte autant que la détection elle-même |

### Conditions de mesure

- **Découper le test set par `parent_ark`, jamais par vue.** Plusieurs vues d'un même
  ouvrage sont quasi identiques : découper par vue garantit une fuite et des scores faux.
- **Nettoyer les labels en amont :** 46 doublons exacts (21 train, 25 val) et 1 boîte
  d'aire nulle (`btv1b21001488-f118`, l.7). Vérifié le 2026-09-17 : aucune autre anomalie
  structurelle sur les 36 687 boîtes, et appariement image/label complet.
- **N'évaluer que la classe `Illustration`.** C'est la seule qu'on livre. Si `Texte` est
  présent à l'entraînement, ses détections sont ignorées au moment de la mesure.
- **Vérifier visuellement les labels avant tout entraînement.** Superposer les boîtes sur
  une trentaine d'images de `train`. Le format est `cx cy w h` normalisé : une confusion
  centre/coin ou une inversion `w`/`h` donne un entraînement qui converge sans rien apprendre.
- **Garder les vues `empty`** au label volontairement vide : ce sont des négatifs purs,
  et la base de la métrique de faux positifs ci-dessus.

---

## 2. Modèles à comparer

Trois entraînements, trois familles.

| Modèle             | Famille       | Pourquoi celui-là                                                                                                                                                             |
| ------------------ | ------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **YOLO26l**        | CNN one-stage | Génération Ultralytics courante (janv. 2026). Apporte **STAL**, une assignation de labels pensée pour les petites cibles — on en a ~1 100. Inférence nativement **NMS-free**. |
| **DocLayout-YOLO** | CNN one-stage | Le seul pré-entraîné sur du **layout documentaire** (DocStructBench) et non sur COCO. Nos images sont des pages, pas des photos naturelles : candidat le plus prometteur.     |
| **RT-DETR-l**      | Transformer   | Famille indépendante, sans NMS par construction. Fort sur les grands objets — et plus de la moitié de nos illustrations couvrent plus de la moitié de la page.                |

**Sans coût supplémentaire :** YOLO26 s'évalue avec `nms=False` **et** `nms=True` sur les
mêmes poids. Répond à une vraie question — le NMS crée-t-il des doublons sur les
illustrations pleine page ?

Licence : Ultralytics et DocLayout-YOLO sont en **AGPL-3.0**, validé pour ce projet.

---

### Une classe ou deux ?

Les labels fournis contiennent deux classes, `Illustration` (30 % des boîtes) et `Texte` (70 %).
On ne livre que la première.

**Réglage principal : `nc=1`**, illustrations seules. C'est la formulation naturelle du
problème. Les blocs de texte ne disparaissent pas des images, ils deviennent du fond :
le modèle apprend « ne pas détecter ici » implicitement, ce qui est le comportement voulu.

**Ablation à mesurer : `nc=2`**, un run unique sur le meilleur modèle. L'hypothèse est que
nommer explicitement le texte aide à le distinguer d'une gravure sur les pages mixtes.
Plausible, mais non démontré — donc on le mesure au lieu de le supposer.

---

## 3. Hyperparamètres

**Identiques pour les trois modèles**, sinon la comparaison ne veut rien dire.

| Paramètre | Valeur           | Motif                                                                                                                              |
| --------- | ---------------- | ---------------------------------------------------------------------------------------------------------------------------------- |
| `imgsz` | 800 | Résolution du corpus. 800 = 32 × 25, compatible avec le stride maximal des modèles. **239 vues (5 %) ne sont pas en 800×800** — surtout des couvertures `-f1`, à leur résolution native : elles seront letterboxées |
| `epochs`  | 50               | Phase de screening                                                                                                                 |
| `batch` | 4 + AMP | À 800 px, une image coûte 1,6× plus qu'à 640. Accumulation de gradient si besoin d'un batch effectif plus large |
| `seed`    | fixée, identique | Comparabilité                                                                                                                      |
| `nc`      | 1                | Illustrations seules ; `nc=2` testé en ablation (cf. §2)                                                                           |

**Augmentations — à revoir avant de lancer.** Les défauts Ultralytics sont pensés pour des
photos naturelles :

- `fliplr` — retourne le texte des pages, douteux sur du document
- `mosaic` — fabrique des mises en page qui n'existent pas dans le corpus

Les deux premiers candidats repartent ensuite pour un schedule complet, à la même
résolution de 800.
