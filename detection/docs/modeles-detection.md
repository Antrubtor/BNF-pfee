# Détection d'illustrations — évaluation et modèle

On détecte les **illustrations** : zéro, une ou plusieurs par vue. Les boîtes produites
alimentent ensuite l'étape de classification, qui ne relève pas de ce document.

Dernière mise à jour : 2026-09-17

---

## 1. Métriques d'évaluation

C'est la partie qui décide. Tant que ces métriques ne sont pas implémentées et figées,
lancer un entraînement ne sert à rien : les résultats ne seraient pas comparables.

### L'IoU, socle de toutes les métriques

L'**IoU** (intersection sur union) entre une boîte prédite et une boîte de vérité terrain
est le **critère d'appariement** de toute la chaîne :

- **AP50** = average precision en comptant correcte toute prédiction dont l'**IoU ≥ 0.50**
- **mAP50-95** = la même, moyennée sur dix seuils d'IoU de 0.50 à 0.95

Soit : **IoU → appariement → comptage TP/FP/FN → précision/rappel → AP**.

**Pourquoi ne pas la rapporter seule.** Elle ne se calcule qu'entre deux boîtes déjà
appariées, donc elle est aveugle aux illustrations manquées et à celles inventées.
Exemple : une page en contient 3, le modèle n'en prédit qu'une avec IoU 0.95.
IoU moyenne 0.95 — excellente ; rappel réel 33 %.

### Métrique principale

**Précision à rappel fixé** (typiquement P@R=0.90).

Le problème métier est l'**hallucination d'illustrations** : le dataset livre 372 vues où
le modèle précédent en a inventé une. On veut donc savoir combien de fausses détections
coûte un niveau de rappel donné.

### Métriques standard

| Métrique | Pourquoi |
|---|---|
| **mAP50-95** | Comparabilité avec la littérature |
| **AP50** | Lecture plus directe de la qualité de détection |
| **Précision / rappel / F1** à seuil fixé | Le point de fonctionnement réel |

### Stratifications — obligatoires

**Par taille de boîte.** La distribution des aires est bimodale : p25 = 3 % de la page,
p75 = 74 %. Les grosses boîtes sont faciles et écrasent la moyenne — **sans stratification,
le régime « petites illustrations » est invisible.**

**Par tag de contenu** (`photographie`, `comic_book`, `film_roll`, `plan`…) : indique sur
quelle sous-population le modèle échoue.

### Métriques spécifiques au problème

| Métrique | Ce qu'elle capture |
|---|---|
| **Taux de faux positifs sur vues sans illustration** | 398 vues n'en contiennent aucune. Mesure directe et lisible de l'hallucination |
| **IoU moyenne des boîtes appariées** | Une boîte trop lâche produit un recadrage inexploitable pour la classification en aval |

### Conditions de mesure

- **Découper le test set par `parent_ark`, jamais par vue.** Plusieurs vues d'un même
  ouvrage sont quasi identiques : découper par vue garantit une fuite.
- **Labels nettoyés en amont** : 22 doublons exacts et 1 boîte d'aire nulle.
  Vérifié le 2026-09-17, aucune autre anomalie sur les 11 051 boîtes `Illustration`.
- **Garder les vues sans illustration** : leur label vide en fait des négatifs purs,
  et la base de la métrique de faux positifs ci-dessus.

---

## 2. Modèle

**YOLO26l.** Génération Ultralytics courante (janvier 2026). Deux de ses apports visent
directement nos difficultés : **STAL**, une assignation de labels pensée pour les petites
cibles — on en a ~1 100 sous 1 % de la page — et l'inférence nativement **NMS-free**.

**Ablation NMS, sans coût.** Le modèle s'évalue avec `nms=False` **et** `nms=True` sur les
mêmes poids entraînés. Répond à une vraie question : le NMS crée-t-il des doublons sur les
illustrations pleine page ?

Licence AGPL-3.0, validé pour ce projet.

---

## 3. Hyperparamètres

| Paramètre | Valeur | Motif |
|---|---|---|
| `imgsz` | 800 | Résolution du corpus, et multiple de 32. **239 vues (5 %) ne sont pas en 800×800** — surtout des couvertures `-f1` : elles seront letterboxées |
| `epochs` | 50 | Premier entraînement |
| `batch` | 6 | Mesuré : 5,1 Go de VRAM à `batch=4` sur les 8,3 Go disponibles. 6 garde une marge |
| `seed` | fixée | Reproductibilité |
| `nc` | 1 | Illustrations seules |

**Une classe ou deux ?** Les labels fournis contiennent aussi une classe `Texte` (70 % des
boîtes), qu'on ne livre pas. On entraîne en `nc=1` : les blocs de texte deviennent du fond,
et le modèle apprend « ne pas détecter ici ». Un run en `nc=2` reste à mesurer en ablation —
avec un handicap connu : **`Texte` n'est annoté que sur 83 % des vues**, donc y entraîner
pénaliserait le modèle sur du texte que personne n'a entouré.

**Augmentations — à revoir avant de lancer.** Les défauts Ultralytics visent des photos
naturelles : `fliplr` retourne le texte des pages, `mosaic` fabrique des mises en page qui
n'existent pas dans le corpus.
