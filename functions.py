"""
Ce fichier "functions.py" regroupe toutes les fonctions nécessaires à la
réalisation d'un MVP pour la société My Content.
Les commentaires et les docstrings sont rédigés en français pour une meilleure
lisibilité. 
"""

# Importation des librairies et modules nécessaires
import logging

import numpy as np
import pandas as pd
import plotly.express as px
from pandas.api.types import is_numeric_dtype
from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler


def articles_feedback(df):
    """
    Calcule le feedback des articles étant donné un jeu de données df_clicks.

    Paramètres
    ----------
    df : Pandas.DataFrame
        Jeu de données contenant les informations sur les clics
        des utilisateurs sur les articles

    Retourne
    -------
    feedback : Pandas.DataFrame
        Jeu de données contenant le feedback des articles pour chaque
        utilisateur
    """

    # Compter le nombre de clics par article et par utilisateur
    article_clicks = df.reset_index().groupby(["user_id", "click_article_id"
                                               ]).agg(clicks_article=("index",
                                                                      "count"))
    user_clicks = df.reset_index().groupby(["user_id"
                                            ]).agg(clicks_user=("index",
                                                                "count"))

    # Joindre le nombre de clics par article et le nombre de clics
    # par utilisateur
    feedback = article_clicks.join(user_clicks, on="user_id")

    # Calcule le feedback pour chaque article
    feedback["feedback"] = (feedback["clicks_article"] /
                            feedback["clicks_user"])

    # Remettre en forme le jeu de données
    feedback = feedback["feedback"].reset_index().rename(
        {"click_article_id": "article_id"}, axis=1)

    return feedback


def clean_df_articles(df: pd.DataFrame, cat_id: int) -> pd.DataFrame:
    """
    Nettoye un DataFrame contenant des articles et leurs caractéristiques.

    Arguments:
        df (pd.DataFrame): DataFrame contenant des articles et leurs
            caractéristiques.
        cat_id (int): ID de la catégorie à conserver.

    Retourne:
        (pd.DataFrame): DataFrame nettoyé.
    """
    # Suppression des colonnes "article_id" et "similarity"
    clean_df = df.drop(["article_id", "similarity"], axis=1, errors="ignore")

    # Remplacement des valeurs de la colonne "category_id" par le "category_id"
    # donné en argument si elles sont égales, sinon par 0
    clean_df["category_id"] = clean_df["category_id"].apply(
        lambda x: cat_id if int(x) == cat_id else 0)

    # Transformation de la colonne "created_at_ts" en valeur int
    clean_df["created_at_ts"] = clean_df["created_at_ts"].apply(
        lambda x: x.value)

    return clean_df


def collaborative_filtering(user_id: int,
                            model,
                            articles_embeddings: pd.DataFrame,
                            num_recommendations: int = 5) -> list:
    """Utilise un modèle de filtrage collaboratif pour trouver
    les articles les plus pertinents pour un utilisateur.

    Arguments:
        user_id {int} -- Identifiant de l'utilisateur.
        model {objet} -- Modèle de filtrage collaboratif.
        articles_embeddings {pd.DataFrame} -- Dataframe avec les articles et
            leurs embeddings.

    Mots-clés:
        num_recommendations {int} -- Nombre d'articles à recommander
            (default: {5})

    Retourne:
        list -- Liste des articles recommandés pour l'utilisateur.
    """

    # Générer une liste des articles et de leurs prédictions
    predictions = [{
        "article_id": prediction.iid,
        "prediction": prediction.est
    } for article in articles_embeddings["article_id"]
        for prediction in [model.predict(uid=user_id, iid=article)]]

    # Trier les articles par prédiction et récupérer les ids
    # des n meilleurs articles
    return list(
        pd.DataFrame(predictions,
                     columns=["article_id", "prediction"]).sort_values(
            by="prediction", ascending=False).reset_index(
            drop=True)["article_id"][:num_recommendations])


def content_based_recommended_articles(popular, articles, result_num=5):
    """
    Cette fonction retourne des articles recommandés basé sur le contenu
    à partir d'un jeu de données donné.

    Arguments:
        popular (pandas DataFrame): Le jeu de données qui contient les
            articles les plus populaires.
        articles (pandas DataFrame): Le jeu de données qui contient
            tous les articles.
        result_num (int): Le nombre de résultats à retourner (défaut: 5).

    Retourne:
        Une liste contenant un DataFrame des articles recommandés,
        un objet StandardScaler,
        un tableau numpy pour les articles et
        un tableau numpy pour les articles populaires.
    """

    # Récupérer l'ID de catégorie de l'article le plus populaire
    category_id = popular["category_id"].iloc[0]

    # Initialiser un objet StandardScaler
    scaler = StandardScaler()

    # Prétraiter les données
    scaled_articles = scaler.fit_transform(
        clean_df_articles(articles, category_id))
    scaled_popular = scaler.transform(clean_df_articles(popular, category_id))

    # Copier le DataFrame des articles
    articles = articles.copy()

    # Calculer la similarité entre les articles populaires
    # et les autres articles
    articles["similarity"] = cosine_similarity(scaled_popular,
                                               scaled_articles)[0]

    # Trier et retourner les articles recommandés
    return articles.sort_values(
        "similarity", ascending=False
    ).iloc[:result_num], scaler, scaled_articles, scaled_popular


def groupby_user_articles(df_articles_features):
    """
    Fonction qui groupe les articles par utilisateur et prend en compte
    les moyennes et les modalités pour chaque colonne.

    Arguments
    ----------
    df_articles_features : DataFrame
       DataFrame contenant les articles et leurs caractéristiques.

    Retourne
    -------
    DataFrame
        DataFrame groupé par utilisateur et contenant les moyennes
        et les modalités pour chaque colonne.
    """
    # définir le type de fonction à appliquer pour chaque colonne
    agg_dict = {}
    for col in df_articles_features.columns:
        if is_numeric_dtype(df_articles_features.dtypes[col]):
            agg_dict[col] = 'mean'
        else:
            agg_dict[col] = lambda x: x.mode()[0]

    # grouper les données par utilisateur
    df_user_articles_features = df_articles_features.groupby(
        lambda x: True).agg(agg_dict)

    return df_user_articles_features


def implicit_article_idxs(user_index: int,
                          rec_model: object,
                          user_interaction_matrix: object,
                          num_recommendations: int = 5) -> list:
    """
    Retourne une liste d'index d'articles pour un utilisateur donné à partir
    du modèle de recommandation et du feedback d'interaction de l'utilisateur.

    Arguments:
        user_index (int): Numero d'index de l'utilisateur
        rec_model (object): Modele utilisé pour la recommandation
        user_interaction_matrix (object): Objet sparse représentant le
            feedback d'interaction de l'utilisateur
        num_recommendations (int, optional): Nombre de recommandations à
            retourner. Par défaut, c'est 5.

    Retourne:
        list: Une liste d'index d'article
    """
    article_indices, _ = rec_model.recommend(
        user_index,
        user_interaction_matrix[user_index],
        N=num_recommendations,
        filter_already_liked_items=False)
    return article_indices


def last_read_articles(user_id: int, df_clicks: pd.DataFrame,
                       df_articles_emb: pd.DataFrame) -> pd.DataFrame:
    """
    Retourne le dernier article lu par un utilisateur donné.

    Arguments:
        user_id (int): l'identifiant de l'utilisateur
        df_clicks (pd.DataFrame): le jeu de données contenant les informations
            sur les clics
        df_articles_emb (pd.DataFrame): le jeu de données contenant les
            informations sur les articles

    Retourne:
        pd.DataFrame: un DataFrame contenant les informations sur le dernier
            article lu par l'utilisateur
    """

    # Convertir l'ID de l'utilisateur en chaîne de caractères
    user_id_str = str(user_id)

    # Récupérer les informations sur le dernier article lu par l'utilisateur
    last_read_article_id = (
        df_clicks.query("user_id == @user_id_str").sort_values(
            "click_timestamp", ascending=False).reset_index(
            drop=True).iloc[0]["click_article_id"])

    # Récupérer les informations sur l'article
    last_read_article_info = df_articles_emb.query(
        "article_id == @last_read_article_id")

    return last_read_article_info


def lightfm_recommendation(user_id: int,
                           model,
                           dataset,
                           num_recommendations: int = 5) -> list:
    """
    Retourne une liste d'`num_recommendations` articles recommandés par le
    modèle `model` pour l'utilisateur `user_id`.

    Arguments
    ----------
    user_id : int
        Identifiant de l'utilisateur pour lequel effectuer les 
        recommandations.
    model : objet
        Modèle utilisé pour calculer les prédictions.
    dataset : objet
        Jeu de données utilisé pour récupérer les articles correspondants 
        aux prédictions.
    num_recommendations : int, optional
        Nombre d'articles à inclure dans la liste de recommandations.
        Par défaut : 5.

    Retourne
    -------
    list
        Une liste des `num_recommendations` articles recommandés pour
        l'utilisateur `user_id`.
    """
    # Récupération de l'index de l'utilisateur
    user_index = dataset._user_id_mapping[user_id]

    # Prédiction des articles recommandés
    predictions = model.predict(user_ids=user_index,
                                item_ids=list(
                                    dataset._item_id_mapping.values()))

    # Création d'un `DataFrame` pour stocker les articles et leurs prédictions
    predictions_df = pd.DataFrame({
        "article_id":
            list(dataset._item_id_mapping.keys()),
        "prediction":
            predictions
    })

    # Tri des articles en fonction de leurs prédictions
    predictions_df.sort_values("prediction", ascending=False, inplace=True)

    # Retourne les `num_recommendations` articles les mieux prédits
    return list(predictions_df.head(num_recommendations)["article_id"])


def plot_tsne_articles(articles_sample_std, popular_articles_std, popular_std,
                       popular_articles, article_embeddings_sample, user_id,
                       user_interest):
    """
    Cette fonction permet de tracer un graphique t-SNE représentant les
    articles recommandés et les articles populaires, ainsi que le niveau
    d'intérêt de l'utilisateur pour la catégorie.

    Arguments:
        articles_sample_std (ndarray): tableau numpy contenant les vecteurs
            des articles recommandés dans leur version standardisée.
        popular_articles_std (ndarray): tableau numpy contenant les vecteurs
            des articles populaires dans leur version standardisée.
        popular_std (ndarray): vecteur du niveau d'intérêt de l'utilisateur
            pour la catégorie dans sa version standardisée.
        popular_articles (DataFrame): DataFrame contenant les articles
            populaires avec leur ID et catégorie.
        article_embeddings_sample (DataFrame): DataFrame avec les ID et
            catégories des articles recommandés.
        user_id (int): ID de l'utilisateur.
        user_interest (DataFrame): DataFrame contenant le niveau d'intérêt de
            l'utilisateur pour la catégorie.

    Retourne:
        (Figures): Graphique t-SNE représentant les articles
            recommandés et les articles populaires, ainsi que le niveau
            d'intérêt de l'utilisateur pour la catégorie.
    """

    # Création du modèle t-SNE
    tsne_model = TSNE(n_components=2)

    # Calcul des vecteurs t-SNE
    articles_tsne = tsne_model.fit_transform(
        np.concatenate(
            (articles_sample_std, popular_articles_std, popular_std)))

    # Séparation des vecteurs t-SNE
    user_interest_tsne = articles_tsne[-1:]
    articles_tsne = articles_tsne[:-1]
    popular_articles_tsne = articles_tsne[-len(popular_articles):]
    articles_tsne = articles_tsne[:-len(popular_articles)]

    # Création du graphique
    fig = px.scatter(x=articles_tsne[:, 0],
                     y=articles_tsne[:, 1],
                     color=article_embeddings_sample["category_id"],
                     symbol=article_embeddings_sample["category_id"],
                     title="TSNE de recommendation d'articles")

    # Ajout des points représentant le niveau d'intérêt de l'utilisateur
    fig.add_scatter(
        x=user_interest_tsne[:, 0],
        y=user_interest_tsne[:, 1],
        mode="markers",
        marker={
            'size': 60,
            'opacity': 0.5
        },
        text=f"Intérêt de l'utilisateur {user_id} sur la catégorie \
        {user_interest['category_id'].iloc[0]}")

    # Ajout des points représentant les articles populaires
    fig.add_scatter(x=popular_articles_tsne[:, 0],
                    y=popular_articles_tsne[:, 1],
                    mode="markers",
                    marker={
                        'color': list(range(len(popular_articles_tsne))),
                        'size': 40,
                        'opacity': 0.5
                    },
                    text=[
                        f"Rang : {i} - Article : {a.article_id} \
                        - Catégorie : {a.category_id}"
                        for i, a in enumerate(popular_articles.itertuples())
                    ])

    # Mise à jour du graphique
    fig.update_layout(showlegend=False)
    fig.show()


def predicted_articles_precision(y_true: dict,
                                 y_pred: dict,
                                 df_articles_emb: pd.DataFrame,
                                 k: int = 5) -> float:
    """Calcule la précision moyenne d'un modèle de recommandation d'articles
        basé sur les articles prédits.

    Arguments:
        y_true {dict}: dictionnaire {user_id: article_id_vrai}
        y_pred {dict}: dictionnaire
            {user_id: [article_id_1, article_id_2, ...]}
        df_articles_emb {pd.DataFrame}: tableau des articles et leurs
            catégories associées
        k {int}: nombre maximum d'articles à prendre en compte pour chaque
            utilisateur (par défaut 5)

    Retourne:
        float: précision moyenne des articles prédits
    """

    # Initialiser la précision moyenne
    average_precision = 0

    # Itérer sur chaque utilisateur
    for user_id, pred_article_ids in y_pred.items():
        # Vérifier si l'utilisateur est présent dans les vraies valeurs
        if user_id not in y_true.keys():
            logging.warning(
                f"L'utilisateur {user_id} est introuvable dans les vraies "
                "valeurs")
            continue

        # Récupérer la catégorie de l'article vrai
        true_category_id = df_articles_emb.iloc[int(
            y_true[user_id])].category_id

        # Récupérer les catégories des articles prédits
        pred_categories = df_articles_emb.iloc[[
            int(pred_article_id) for pred_article_id in pred_article_ids[:k]
        ]].category_id

        # Calculer la précision moyenne
        average_precision += len(
            pred_categories[pred_categories == true_category_id]) / k

    # Diviser par le nombre d'utilisateurs
    return average_precision / len(y_pred)


def read_articles(user_id: int, clicks_df: pd.DataFrame,
                  articles_embeddings_df: pd.DataFrame) -> pd.DataFrame:
    """
    Cette fonction lit les articles lus par un utilisateur et retourne
    un dataframe des articles les plus populaires de cet utilisateur.

    Arguments:
        user_id (int): l'ID d'utilisateur.
        clicks_df (pd.DataFrame): le dataframe des clics par utilisateur.
        articles_embeddings_df (pd.DataFrame): le dataframe des articles et de
            leurs embeddings.

    Retourne:
        pd.DataFrame: Le dataframe des articles les plus populaires de cet
            utilisateur.
    """
    user_clicked_articles = clicks_df[clicks_df['user_id'] == str(
        user_id)]['click_article_id']
    popular_articles = groupby_user_articles(
        articles_embeddings_df[articles_embeddings_df['article_id'].isin(
            user_clicked_articles)]).drop(['article_id'], axis=1)
    return popular_articles


def recommendation_metrics(real_values: dict,
                           predictions: dict,
                           method: str = 'mean') -> float:
    """
    Calcule la métrique de recommandation en fonction de la méthode choisie.

    Arguments
    ----------
    real_values : dict
        Dictionnaire des vraies valeurs
    predictions : dict
        Dictionnaire des prédictions
    method : str, optional
        La méthode à utiliser pour le calcul de la métrique, par défaut 'mean'

    Retourne
    -------
    float
        La métrique de recommandation calculée
    """

    metric_score = 0
    total_user_count = 0

    for user_id, predicted_article_ids in predictions.items():
        # Vérifie que l'utilisateur est bien présent dans les vraies valeurs
        if user_id not in real_values.keys():
            logging.info(
                f"L'utilisateur {user_id} est introuvable dans les vraies valeurs."
            )
            continue

        # Récupère la valeur réelle de l'utilisateur
        true_article_id = str(real_values[user_id])

        # Vérifie que l'article est bien présent dans les prédictions
        if true_article_id not in predicted_article_ids:
            logging.info(
                f"L'article {true_article_id} n'a pas été trouvé dans les "
                "prédictions de l'utilisateur {user_id}.")
            continue

        # Calcule le classement de l'article
        article_rank = predicted_article_ids.index(true_article_id) + 1

        # Calcule la métrique en fonction de la méthode choisie
        if method == 'mean':
            metric_score += article_rank
        elif method == 'score':
            metric_score += 1 / article_rank
        else:
            raise ValueError(f"La méthode '{method}' n'est pas supportée.")

        total_user_count += 1

    return metric_score / total_user_count


def simple_cbf_recommendation(user_id: int,
                              interactions_df: pd.DataFrame,
                              article_embeddings: np.ndarray,
                              number_of_recommendations: int = 5) -> list[int]:
    """Renvoie les n articles les plus similaires à l'article le plus
    récent lu par l'utilisateur.

    Arguments
    ----------
    user_id : int
        Identifiant de l'utilisateur
    interactions_df : pd.DataFrame
        DataFrame contenant les interactions entre les utilisateurs et les
        articles
    article_embeddings : np.ndarray
        Embeddings à utiliser pour calculer la similarité entre les articles
    number_of_recommendations : int
        Nombre d'articles à recommander

    Retourne
    -------
    list[int]
        Liste des identifiants des articles recommandés
    """
    # Convertit les colonnes numériques en entiers
    interactions_df = interactions_df.astype(np.int64)
    # Convertit les embeddings en flottant
    embeddings_for_recommendation = article_embeddings.astype(np.float32)
    # Récupère la liste des articles lus par l'utilisateur
    user_read_articles = interactions_df.loc[interactions_df.user_id ==
                                             user_id]['article_id'].to_list()
    # Récupère l'identifiant de l'article le plus récent lu
    most_recent_article = user_read_articles[-1]
    # Supprime les articles précédemment lus de la liste des embeddings
    for article_id in user_read_articles[:-1]:
        embeddings_for_recommendation = np.delete(
            embeddings_for_recommendation, [article_id], 0)
    # Supprime l'article précédemment lu avant de calculer les similarités
    embeddings_without_last_read_article = np.delete(
        embeddings_for_recommendation, [most_recent_article], 0)
    # Calcule les similarités entre l'article le plus récent lu et les autres
    articles_similarity = cosine_similarity(
        [embeddings_for_recommendation[most_recent_article]],
        embeddings_without_last_read_article)[0]
    # Récupère les n articles les plus similaires
    recommended_articles_ids = np.argsort(
        articles_similarity)[::-1][0:number_of_recommendations]
    # Renvoie la liste des identifiants des articles recommandés
    return recommended_articles_ids
