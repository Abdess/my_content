# My Content

![My Content](data/images/logo.png)

---

**Lire le notebook en ligne nbviewer** : [![nbviewer](https://raw.githubusercontent.com/jupyter/design/master/logos/Badges/nbviewer_badge.svg)](https://nbviewer.jupyter.org/github/Abdess/my_content/blob/main/MyContent.ipynb)

**Lire la documentation en ligne nbviewer** : [![nbviewer](https://raw.githubusercontent.com/jupyter/design/master/logos/Badges/nbviewer_badge.svg)](https://nbviewer.jupyter.org/github/Abdess/my_content/blob/main/data/doc/functions.html)

---

**My Content** est une start-up qui a pour objectif d'encourager la lecture en proposant des contenus pertinents à ses utilisateurs. En tant que CTO, notre mission est de créer un premier MVP qui offrira une solution innovante et fiable de recommandation de contenus.

## Mission

Notre mission est de créer une solution innovante et fiable de recommandation de contenus qui émerveillera les utilisateurs de **My Content**. Pour cela, nous allons utiliser le jeu de données "[News Portal User Interactions by Globo.com](https://www.kaggle.com/datasets/gspmoreira/news-portal-user-interactions-by-globocom)", l'explorer et réfléchir aux meilleures approches pour le concevoir. Nous allons ensuite le déployer sur le cloud et l'intégrer à l'interface streamlit de **My Content** afin de donner à ses utilisateurs une expérience inoubliable.

## Outils et technologies utilisés

Pour réaliser ce projet, nous avons utilisé les bibliothèques suivantes :

- **cosmos** : pour intégrer les données dans Azure Cosmos DB.
- **dotenv** : pour charger les variables d'environnement.
- **libreco** : pour implémenter un système de recommandation basé sur des algorithmes de collaboration.
- **lightfm** : pour implémenter un système de recommandation basé sur des algorithmes factorisés.
- **scipy** : pour transformer les données en matrice creuse.
- **sklearn** : pour l'encodage des données et la décomposition en composantes principales.
- **surprise** : pour implémenter un système de recommandation basé sur des algorithmes matriciels.
- **implicit** : pour implémenter un système de recommandation basé sur des algorithmes de matrices creuses.

## Résultats

À l'heure actuelle, nous avons réussi à mettre en œuvre un MVP qui offre des recommandations d'articles à nos utilisateurs. Nous avons déployé avec succès cette solution sur un environnement cloud. Le résultat du MVP est hébergé sur streamlit à cette adresse : <https://abdess-my-content-streamlit-app-aafs5i.streamlit.app/>

![MVP](data/images/Streamlit.png)

## Sources

Les codes sources du projet sont disponibles dans les dépôts suivant :

- Les algorithmes de recommandation : <https://github.com/Abdess/my_content>
- Azure functions: <https://github.com/Abdess/my_content_azure>
- MVP: <https://github.com/Abdess/my_content_streamlit>
