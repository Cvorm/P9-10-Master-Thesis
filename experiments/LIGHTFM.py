# from generic_preprocessing import *
# from IPython.display import HTML
import pandas as pd
import numpy as np
from scipy import sparse
from lightfm import LightFM
from sklearn.metrics.pairwise import cosine_similarity
from lightfm.evaluation import precision_at_k
from lightfm.cross_validation import random_train_test_split
import scipy
from scipy.sparse import coo_matrix
from lightfm.evaluation import auc_score
from lightfm.evaluation import precision_at_k
from lightfm.evaluation import recall_at_k
from engine import evaluation
from engine.evaluation import recommender_precision,recommender_recall,novelty

def run_lightfm(movies, ratings, train, test, k_items):
    def create_interaction_matrix(df,user_col, item_col, rating_col, norm= False, threshold = None):
        '''
        Function to create an interaction matrix dataframe from transactional type interactions
        Required Input -
            - df = Pandas DataFrame containing user-item interactions
            - user_col = column name containing user's identifier
            - item_col = column name containing item's identifier
            - rating col = column name containing user feedback on interaction with a given item
            - norm (optional) = True if a normalization of ratings is needed
            - threshold (required if norm = True) = value above which the rating is favorable
        Expected output -
            - Pandas dataframe with user-item interactions ready to be fed in a recommendation algorithm
        '''
        interactions = df.groupby([user_col, item_col])[rating_col] \
                .sum().unstack().reset_index(). \
                fillna(0).set_index(user_col)
        if norm:
            interactions = interactions.applymap(lambda x: 1 if x > threshold else 0)
        return interactions

    def create_user_dict(interactions):
        '''
        Function to create a user dictionary based on their index and number in interaction dataset
        Required Input -
            interactions - dataset create by create_interaction_matrix
        Expected Output -
            user_dict - Dictionary type output containing interaction_index as key and user_id as value
        '''
        user_id = list(interactions.index)
        user_dict = {}
        counter = 0
        for i in user_id:
            user_dict[i] = counter
            counter += 1
        return user_dict


    def create_item_dict(df,id_col,name_col):
        '''
        Function to create an item dictionary based on their item_id and item name
        Required Input -
            - df = Pandas dataframe with Item information
            - id_col = Column name containing unique identifier for an item
            - name_col = Column name containing name of the item
        Expected Output -
            item_dict = Dictionary type output containing item_id as key and item_name as value
        '''
        item_dict ={}
        for i in range(df.shape[0]):
            item_dict[(df.loc[i,id_col])] = df.loc[i,name_col]
        return item_dict


    def runMF(interactions, n_components=30, loss='warp', k=15, epoch=30,n_jobs = 4):
        '''
        Function to run matrix-factorization algorithm
        Required Input -
            - interactions = dataset create by create_interaction_matrix
            - n_components = number of embeddings you want to create to define Item and user
            - loss = loss function other options are logistic, brp
            - epoch = number of epochs to run
            - n_jobs = number of cores used for execution
        Expected Output  -
            Model - Trained model
        '''
        x = sparse.csr_matrix(interactions.values)
        model = LightFM(no_components= n_components, loss=loss,k=k)
        model.fit(x, epochs=epoch,num_threads = n_jobs)
        return model

    np.random.seed(1)
    interactions = create_interaction_matrix(df=ratings,
                                             user_col='userId',
                                             item_col='movieId',
                                             rating_col='rating')

    train_interactions = create_interaction_matrix(df=train,
                                             user_col='userId',
                                             item_col='movieId',
                                             rating_col='rating')

    test_interactions = create_interaction_matrix(df=test,
                                             user_col='userId',
                                             item_col='movieId',
                                             rating_col='rating')

    user_dict = create_user_dict(interactions=interactions)
    movies_dict = create_item_dict(df=movies,
                                   id_col='movieId',
                                   name_col='title')

    mf_model = runMF(interactions=interactions, # should be train
                     n_components=30,
                     loss='warp',
                     epoch=30,
                     n_jobs=4)
    train = sparse.csr_matrix(train_interactions.values)
    test = sparse.csr_matrix(test_interactions.values)
    train = train.tocoo()
    test = test.tocoo()

    train_auc = auc_score(mf_model, test).mean()

    print('Hybrid training set AUC: %s' % train_auc)

    trainprecision = precision_at_k(mf_model, test, k=k_items).mean()

    print('Hybrid training set precision: %s' % trainprecision)

    trainrecall = recall_at_k(mf_model, test, k=k_items).mean()
    print('Hybrid training set recall: %s' % trainrecall)

    n_users, n_items = interactions.shape # no of users * no of items
    pred_df = pd.DataFrame(columns=np.unique(ratings.movieId), index=np.arange(n_users))
    for uid in pred_df.T.columns:
        bobs = mf_model.predict(uid, np.arange(n_items))
        temp= bobs.tolist()
        pred_df.loc[uid] = temp
    pred_df = pred_df.T
    pred_list = []
    test_list = []
    seen_list = []
    for column in pred_df:
            user = pred_df[column]
            sorted = user.sort_values(ascending=False)
            pred_movies = list(sorted.index)
            pred_list.append(pred_movies[:k_items])
    xu_train_df = pd.DataFrame(train.toarray(), columns=np.unique(ratings.movieId), index=np.arange(n_users))
    xu_train_df = xu_train_df.T
    xu_test_df = pd.DataFrame(test.toarray(), columns=np.unique(ratings.movieId), index=np.arange(n_users))
    xu_test_df = xu_test_df.T
    for column in xu_test_df:
            user_test = xu_test_df[column]
            filtered = user_test.where(user_test > 0)
            # true_movies = list(filtered.index)
            true_movies = user_test[user_test > 0]
            true_movie_ids = list(true_movies.index)
            test_list.append(true_movie_ids)
    precision = recommender_precision(pred_list, test_list)
    recall = recommender_recall(pred_list, test_list)
    # niggerty = novelty(pred_df.T, xu_train_df.T, np.unique(ratings.userId), np.unique(ratings.movieId), 5)
    print(precision, recall) # niggerty

# ratings = pd.read_csv('../Data/ratings_100k.csv', sep=',')
# movies = pd.read_csv('../Data/movie_new.csv', sep=',')
# run_lightfm(movies,ratings)