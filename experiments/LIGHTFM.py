# from generic_preprocessing import *
# from IPython.display import HTML
from collections import defaultdict
import pandas as pd
import numpy as np
from scipy import sparse
from lightfm import *
from lightfm.data import Dataset
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

def run_lightfm(movies, ratings, train, test, k_items, train_df_split):

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


    def runMF(interactions, item_features, n_components=30, loss='warp', k=15, epoch=30,n_jobs = 4):
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
        model.fit(x, item_features=item_features, epochs=epoch,num_threads = n_jobs)
        return model

    np.random.seed(1)
    #yo = Dataset.build_item_features()
    #
    # interactions = create_interaction_matrix(df=ratings,
    #                                          user_col='userId',
    #                                          item_col='movieId',
    #                                          rating_col='rating')
    #
    # train_interactions = create_interaction_matrix(df=train,
    #                                          user_col='userId',
    #                                          item_col='movieId',
    #                                          rating_col='rating')
    #
    test_interactions = create_interaction_matrix(df=test,
                                             user_col='userId',
                                             item_col='movieId',
                                             rating_col='rating')
    #
    # user_dict = create_user_dict(interactions=interactions)
    # movies_dict = create_item_dict(df=movies,
    #                                id_col='movieId',
    #                                name_col='title')

    # built_dif.fit((x for x in train['userId']),
    #               (x for x in train['movieId']),
    #               item_features=(x for x in movies['rating'].values))
    dataset = pd.DataFrame(columns=['movieId', 'rating'])
    dataset['movieId'] = movies['movieId']
    movies['rating'] = movies['rating'].round()
    dataset['rating'] = 'rating_' + movies['rating'].astype(int).astype(str)

    item_ids = np.unique(train.movieId.astype(int))
    item_features_list = [f'rating_{f}' for f in range(11)]
    item_features = [(int(x['movieId']), [x['rating']]) for y, x in dataset.iterrows()]
    user_ids = np.unique(train.userId)
    built_dif = Dataset()
    built_dif.fit_partial(users=user_ids)
    built_dif.fit_partial(items=item_ids)
    built_dif.fit_partial(item_features=item_features_list)
    dataset_item_features = built_dif.build_item_features(item_features)
    (interactions, weights) = built_dif.build_interactions(((x['userId'], x['movieId']) for y, x in train.iterrows()))
    modelx = LightFM(no_components=30, loss='bpr', k=15, random_state=1)
    modelx.fit(interactions, epochs=30, num_threads=4, item_features=dataset_item_features) #item_features=dataset_item_features
    test = sparse.csr_matrix(test_interactions.values)
    test = test.tocoo()
    num_users, num_items = built_dif.interactions_shape()
    print('Num users: {}, num_items {}.'.format(num_users, num_items))
    trainprecision = precision_at_k(modelx, test, k=k_items, item_features=dataset_item_features).mean() #item_features=dataset_item_features,
    print('Hybrid training set precision: %s' % trainprecision)
    trainrecall = recall_at_k(modelx, test, k=k_items, item_features=dataset_item_features).mean() #item_features=dataset_item_features
    print('Hybrid training set recall: %s' % trainrecall)
    # movies = pd.concat([movies_test, movies_train])
    # features = [(x['movieId'], x['rating']) for y, x in movies.iterrows()]
    # features1 = [(x['movieId'], x['genres'].split(', ')) for y, x in movies.iterrows()]
    # built_dif.fit(
    #     (x for x in train['userId']),
    #     # (x for x in train['movieId']),
    #     items=(x for x in train['movieId'])
    #     # item_features=(x for x in movies['rating'])
    # )
    # num_users, num_items = built_dif.interactions_shape()
    # print('Num users: {}, num_items {}.'.format(num_users, num_items))
    # # built_dif.fit_partial(# items=(x for x in movies['movieId']),
    # #                     item_features=(x for x in movies['rating']))
    # num_users, num_items = built_dif.interactions_shape()
    # print('Num users: {}, num_items {}.'.format(num_users, num_items))
    # item_features = built_dif.build_item_features(((x['movieId'], ['rating']) #[x['rating']]
    #                                               for y, x in movies.iterrows())
    #                                               , normalize=True) #features
    # built_dif.fit_partial(item_features=item_features)
    # (interactions, weights) = built_dif.build_interactions(((x['userId'], x['movieId']) for y, x in train.iterrows()))
    # test = sparse.csr_matrix(test_interactions.values)
    # test = test.tocoo()
    # num_users, num_items = built_dif.interactions_shape()
    # print('Num users: {}, num_items {}.'.format(num_users, num_items))
    # modelx = LightFM(no_components=30, loss='warp', k=15)
    # modelx.fit(interactions, item_features=item_features, epochs=30, num_threads=4) #
    # mf_model = runMF(interactions=inter, # should be train
    #                  item_features=item_features,
    #                  n_components=30,
    #                  loss='warp',
    #                  epoch=30,
    #                  n_jobs=4)


    n_users, n_items = interactions.shape # no of users * no of items
    pred_df = pd.DataFrame(columns=item_ids, index=user_ids) #np.arange(n_users))
    for uid in pred_df.T.columns:
        bobs = modelx.predict(uid - 1, np.arange(n_items), item_features=dataset_item_features) #item_features=item_features
        temp = bobs.tolist()
        pred_df.loc[uid] = temp
    pred_df = pred_df.T
    pred_list = []
    pred_dict = defaultdict(list)
    test_list = []
    for column in pred_df:
            user = pred_df[column]
            sorted = user.sort_values(ascending=False)
            pred_movies = list(sorted.index)
            pred_list.append(pred_movies[:k_items])
            pred_dict[column] = pred_movies[:k_items]
    xu_train_df = pd.DataFrame(interactions.toarray(), columns=np.unique(ratings.movieId), index=np.arange(n_users))
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
    niggerty = novelty(pred_dict, train_df_split, np.unique(ratings.userId), np.unique(ratings.movieId), 5) #pred_df.T
    print(precision, recall, niggerty)

# ratings = pd.read_csv('../Data/ratings_100k.csv', sep=',')
# movies = pd.read_csv('../Data/movie_new.csv', sep=',')
# run_lightfm(movies,ratings)