import sys
import random
import pandas as pd
import jsonlines
import numpy as np
import copy
import pickle
import json
random.seed(42)
np.random.seed(42)
dataset_names = ['ml-1m']
isHint = True
sample_method = 'uniform'  

#load info for kg
output_df = pd.read_csv('output22.csv', sep='|', header=None, names=['movie_id', 'movie_name', 'encoding'])
output_movie_id_types = output_df['movie_id'].apply(type).unique()
entity_df = pd.read_csv('only_entity-id.tsv', sep='\t', header=None, names=['encoding', 'translate1', 'translate2', 'translate3', 'entity_id'])
#kg_less_test_df = pd.read_csv('selected_triplets_text.tsv', sep='\t', header=None, names=['head', 'relation', 'tail'])
#kg_less_test_id_df = pd.read_csv('selected_triplets_id.tsv', sep='\t', header=None, names=['head_id', 'relation_id', 'tail_id'])
kg_less_test_df = pd.read_csv('pretrain-output_kg_text.tsv', sep='\t', header=None, names=['head', 'relation', 'tail'])
kg_less_test_id_df = pd.read_csv('pretrain-output_kg_id.tsv', sep='\t', header=None, names=['head_id', 'relation_id', 'tail_id'])
kg_head_id_types = kg_less_test_id_df['head_id'].apply(type).unique()
kg_tail_id_types = kg_less_test_id_df['tail_id'].apply(type).unique()
relations_df = pd.read_csv('all_relations_id.tsv', sep='\t', header=None, names=['relation_id', 'relation'])
movie_name_dict = dict(zip(output_df['movie_id'], output_df['movie_name']))
movie_id_to_encoding = dict(zip(output_df['movie_id'], output_df['encoding']))
entity_id_to_name = dict(zip(entity_df['entity_id'], entity_df['translate1']))
relation_id_to_description = dict(zip(relations_df['relation_id'], relations_df['relation']))

#load triples for like items
def get_kg_triples_like_1(movie_list, kg_id_df, kg_text_df, entity_to_name, relation_to_desc):
    triples = []
    
    for movie in movie_list:
        related_triples = kg_id_df[(kg_id_df['head_id'] == movie) | (kg_id_df['tail_id'] == movie)]
        
        if not related_triples.empty:
            selected_triples = related_triples.sample(n=min(1, len(related_triples)), random_state=42)
            
            for _, triple in selected_triples.iterrows():
                head_text = kg_text_df.iloc[triple.name]['head']
                relation_text = kg_text_df.iloc[triple.name]['relation']
                tail_text = kg_text_df.iloc[triple.name]['tail']
                #triples.append(f'{head_text} {relation_text} {tail_text}')
                triples.append(f'{head_text} - {relation_text} - {tail_text}')
    
    return triples

#load triples for candidate items
def get_kg_triples_wait_1(movie_list, kg_id_df, kg_text_df, entity_to_name, relation_to_desc):
    triples = []
    
    for movie in movie_list:
        related_triples = kg_id_df[(kg_id_df['head_id'] == movie) | (kg_id_df['tail_id'] == movie)]
        
        if not related_triples.empty:
            selected_triples = related_triples.sample(n=min(1, len(related_triples)), random_state=42)
            
            for _, triple in selected_triples.iterrows():
                head_text = kg_text_df.iloc[triple.name]['head']
                relation_text = kg_text_df.iloc[triple.name]['relation']
                tail_text = kg_text_df.iloc[triple.name]['tail']
                #triples.append(f'{head_text} {relation_text} {tail_text}')
                triples.append(f'{head_text} - {relation_text} - {tail_text}')
    
    return triples

def sort_list_reverse_with_indices(lst):
    sorted_indices = sorted(enumerate(lst), key=lambda x: x[1], reverse=True)
    sorted_indices = [index for index, _ in sorted_indices]
    return sorted_indices


def load_user_ids_from_candidate_items(file_path):
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        user_ids = list(data.keys())
        print(f" Loaded {len(user_ids)} user IDs from {file_path}")
        return user_ids
    except Exception as e:
        print(f" Error loading user IDs from {file_path}: {e}")
        return []


def load_representative_items(file_path):
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        print(f" Loaded representative items for {len(data)} users from {file_path}")
        return data
    except Exception as e:
        print(f" Error loading representative items from {file_path}: {e}")
        return {}


def load_candidate_items(file_path):
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        print(f" Loaded candidate items for {len(data)} users from {file_path}")
        return data
    except Exception as e:
        print(f" Error loading candidate items from {file_path}: {e}")
        return {}


def get_user_representative_items(user_id, representative_data):
    
    try:
       
        user_key = str(user_id)
        user_representative_data = representative_data.get(user_key, {})
        representative_movies = user_representative_data.get('representative_movies', [])
        
        
        movie_ids = []
        ratings = []
        for movie in representative_movies:
            movie_ids.append(movie['movie_id'])
            ratings.append(movie['rating'])
        
        return movie_ids, ratings
    except Exception as e:
        print(f" Error getting representative items for user {user_id}: {e}")
        return [], []


def get_user_candidate_items_sorted(user_id, candidate_data):
    
    try:
        
        user_key = str(user_id)
        user_candidate_data = candidate_data.get(user_key, {})
        candidate_movies = user_candidate_data.get('candidate_movies', [])
        
        if len(candidate_movies) < 5:
            print(f" User {user_id} has only {len(candidate_movies)} candidate items, need at least 5")
            return []
        
       
        sorted_candidates = sorted(candidate_movies, key=lambda x: x['rating'], reverse=True)
        
        
        movie_ids = [movie['movie_id'] for movie in sorted_candidates]
        
        return movie_ids
    except Exception as e:
        print(f" Error getting candidate items for user {user_id}: {e}")
        return []

#load data
for dataset_name in dataset_names:
    if dataset_name == 'ml-1m':
        df_like = pd.read_csv('./train_set.txt', names=['u', 'i', 'r', 't'], sep=' ')
        df_dislike = pd.read_csv('./dislike.txt', header=None, names=['u', 'i', 'r', 't'])
        movie_info = pd.read_csv('./movie_info.csv', header=None,names=['movie_id', 'movie_name', 'year', 'genre'], sep='|', engine='python', encoding='latin-1')
        df_like_p = pd.read_csv('./train_set.txt', sep=' ')
        df_like_p.columns = ['u', 'i', 'r', 't']
        print(df_like_p.head())
        movie_id_list = movie_info['movie_id'].tolist()
        movie_name_list = movie_info['movie_name'].tolist()
        movie_name_dict = {movie_id_list[i]: movie_name_list[i] for i in range(len(movie_name_list))}
    
    mes_list_pointwise = []
    mes_list_pairwise = []
    mes_list_pairwise_inv = []
    mes_list_listwise = []
    if 'book' in dataset_name:
        kws = 'book'
    else:
        kws = 'movie'


    def sort_list_reverse_with_indices(lst):
        sorted_indices = sorted(enumerate(lst), key=lambda x: x[1], reverse=True)
        sorted_indices = [index for index, _ in sorted_indices]
        return sorted_indices

    import pickle

    with open('./item.pkl', "rb") as file:
        cm_item = pickle.load(file)
    with open('./user.pkl', "rb") as file:
        cm_user = pickle.load(file)
    with open('./pred.pkl', "rb") as file:
        cm_pred = pickle.load(file)    
    with open('./item_id_mapping.pkl', "rb") as file:
        mf_item = pickle.load(file)
    #with open('./item_id_mapping-all.pkl', "rb") as file:
        #all_item = pickle.load(file)
    with open('./user_id_mapping.pkl', "rb") as file:
        mf_user = pickle.load(file)
    with open('./rating_matrix.pkl', "rb") as file:
        mf_pred = pickle.load(file)
    with open('./user.pkl', "rb") as file:
        cm_user_emb = pickle.load(file)
  
    mes_list = []
    gt_list = []


    
    print(" Loading user IDs from candidate_items.json...")
    sample_list = load_user_ids_from_candidate_items('candidate_items.json')
    
    if not sample_list:
        print(" No user IDs found in candidate_items.json, exiting...")
        sys.exit(1)
    
    print(f" Using {len(sample_list)} users from candidate_items.json")
    print(f" Sample user IDs: {sample_list[:10]}...")  
    
    
    print(" Loading representative items from representative_items.json...")
    representative_data = load_representative_items('representative_items.json')
    
    if not representative_data:
        print(" No representative items found, exiting...")
        sys.exit(1)
    
    
    print("📊 Loading candidate items from candidate_items.json...")
    candidate_data = load_candidate_items('candidate_items.json')
    
    if not candidate_data:
        print("❌ No candidate items found, exiting...")
        sys.exit(1)
    

    """
    if 'ml-1M' in dataset_name:
        sample_n = 1000
    else:
        sample_n = 1000
    user_list = list(df_like['u'].unique())
    sample_list = []
    import math

    weights = [math.log(len(df_like[df_like['u'] == uni])) for uni in user_list]


    if sample_method == 'uniform':
        for i in range(sample_n):
            sample_ = random.sample(user_list, 1)[0]
            sample_list.append(sample_)
    else:
        sample_list1 = []
        sample_list2 = []

        sample_imp = int(sample_n * 0.6)
        
        #1
        user_ids = sorted(cm_user_emb.keys(), key=int)
        cm_user_emb_matrix = np.array([cm_user_emb[user] for user in user_ids])
        weights = [math.log(len(df_like[df_like['u'] == int(user)])) for user in user_ids]

        
        for i in range(sample_imp):
            sample_ = random.choices(user_list, weights, k=1)[0]
            sample_list1.append(sample_)

        from sklearn.cluster import KMeans

        kmeans = KMeans(n_clusters=10, random_state=0).fit(cm_user_emb_matrix)


        
        labels = kmeans.labels_

        counts = np.bincount(labels)

        samples_per_cluster = np.round(counts / counts.sum() * sample_imp).astype(int)

        sampled_ids = []
        for cluster_id, samples in enumerate(samples_per_cluster):

            cluster_indices = np.where(labels == cluster_id)[0]
            cluster_users = [user_ids[i] for i in cluster_indices] 
            sampled_ids.extend(np.random.choice(cluster_users, samples, replace=True))
            


        sample_list1.extend(sampled_ids)
        from collections import Counter

        occurrences = Counter(sample_list1)
        t_occurrences = {element: 0.95 ** (count - 1) for element, count in occurrences.items()}
        sample_list2 = [t_occurrences[_] for _ in sample_list1]

        sample_list = random.choices(sample_list1, weights=sample_list2, k=sample_n)
    """


    processed_users = 0
    skipped_users = 0
    
    for uni in sample_list:
        
        uni_int = int(uni)
        df = df_like[df_like['u'] == uni_int]
        df_un = df_dislike[df_dislike['u'] == uni_int]

        if len(df) > 1:
            processed_users += 1

            '''
            Pointwise Ranking
            '''
            dfp = df_like_p[df_like_p['u'] == uni_int]

            my_list = dfp['i'].tolist()
            my_list_r = dfp['r'].tolist()


            rndl = [i_ for i_ in range(len(my_list))]
            random.shuffle(rndl)

            try:
                if not 'book' in dataset_name:
                    my_list = [int(my_list[x]) for x in rndl]
                    my_list_r = [int(my_list_r[x]) for x in rndl]
                else:
                    my_list = [(my_list[x]) for x in rndl]
                    my_list_r = [(my_list_r[x]) for x in rndl]
            except Exception as e:
                print(f"{e}")
            if len(dfp) > 50:
                topk = 50
            else:

                topk = max(5, len(dfp) - 3)  
            trainlist = my_list[:topk]
            trainlist_r = my_list_r[:topk]

            testlist = my_list[-1:]
            if not testlist:
                print(f"Skipping user {uni_int} (Pointwise): testlist empty, my_list length: {len(my_list)}")
                continue  

            testlist_r = my_list_r[-1:]

            yy = mf_item.get(str(testlist[0]), None)
            uu = mf_user.get(str(uni_int), None)

            
            
            if yy is not None and uu is not None:
                try:
                    mf_lable = mf_pred[uu][yy]
                except IndexError:
                    mf_lable = 'Unknown.' 
                except Exception as e:
                    mf_lable = 'Unknown.'
            else:
                mf_lable = 'Unknown.'
            
            historical_interactions = [f'"{movie_name_dict[i]}"' for i in trainlist]
            answer_items_set = [f'"{movie_name_dict[i]}"' for i in testlist]

            historical_interactions = [historical_interactions[i_] + ': ' + str(trainlist_r[i_]) + ';' for i_ in
                                       range(len(historical_interactions))]

            historical_interactions = ' '.join(historical_interactions)
            if 'book' in dataset_name:
                highest_score = 5
            else:
                highest_score = 5
            instruct0 = f'''You are a {kws} recommender system. Your task is to predict the relevance score to a target {kws} based on the user's historical {kws} ratings. The score should be between 1 and {highest_score}, where 1 is the lowest affinity and {highest_score} is the highest. Respond only with a number between 1 to {highest_score}.\n\n'''

            instruct1 = f'''User's historical {kws} ratings: <historical interactions>. \n\nQuestion: Based on the user's historical ratings, predict the relavance score of the target {kws} <movie> with the user.\n'''
            instruct2 = '''Hint: Another recommender system suggests the answer is <mf_prediction>"'''
            instruct3 = '\n\nPlease only output the score.\n'
            if isHint:
                instruct1 = instruct1 + instruct2
            instruct1 = instruct1.replace('<historical interactions>', historical_interactions).replace('<movie>', answer_items_set[0]).replace('<mf_prediction>', str(mf_lable)[:3])

            fi = {'messages': [
                {"role": "system", "content": [{"type": "text", "content": ""}]},
                {"role": "user", "content": [{"type": "text", "content": instruct0 + instruct1 + instruct3}]},
                {"role": "assistant", "content": [{"type": "text", "content": 'Answer: ' + str(testlist_r[0])}]}
            ]}
            mes_list.append(fi)
            mes_list_pointwise.append(fi)

            '''
            Pairwise Ranking
            '''
            unlikelist = []
            if len(df_un) > 0:
                my_list = df_un['i'].tolist()
                random.shuffle(my_list)

                if not 'book' in dataset_name:
                    my_list = [int(x) for x in my_list]
                else:
                    my_list = [(x) for x in my_list]

                unlikelist = my_list[:10]

            my_list = df['i'].tolist()
            random.shuffle(my_list)

            if not 'book' in dataset_name:
                my_list = [int(x) for x in my_list]
            else:
                my_list = [(x) for x in my_list]

            if len(df) > 55:
                topk = 50
            else:
                topk = len(df) - 3
            trainlist = my_list[:topk]
            testlist = my_list[-1:]

            if not trainlist:
                print(f"Skipping user {uni_int} (Pairwise): trainlist empty, df length: {len(df)}, topk: {topk}")
                continue  
                
            neglist = []
            while len(neglist) < 1:
                rn = random.sample(movie_id_list, 1)[0]
                if not rn in my_list:
                    neglist.append(rn)

            random_n = (random.random() > 0.5)

            
            
            historical_interactions = [f'"{movie_name_dict[i]}"' for i in trainlist]

            false_items_set = [f'"{movie_name_dict[i]}"' for i in neglist]

            answer_items_set = [f'"{movie_name_dict[i]}"' for i in testlist]



            sample_cm_item_keys = list(cm_item.keys())[:10]
            
            try:
                xx = cm_item[str(neglist[0])]
                yy = cm_item[str(testlist[0])]
                uu = cm_user.get(str(uni_int), None)
                if xx is not None and yy is not None and uu is not None:
                    if cm_pred.get((str(uu), str(yy)), 0) > cm_pred.get((str(uu), str(xx)), 0):
                        cm_lable = 'Yes.'
                    else:
                        cm_lable = 'No.'
                else:
                    cm_lable = 'Unknown.'
            except IndexError as e:
                print(f" IndexError: {e}")
                cm_lable = 'Unknown.'
            except Exception as e:
                cm_lable = 'Unknown.'

            unlikelist = [x_ for x_ in unlikelist if x_ in movie_name_dict.keys()]

            user_unpre = [f'"{movie_name_dict[i]}"' for i in unlikelist]

            if len(unlikelist) < 3:
                user_unpre = 'None.'
            else:
                user_unpre = ', '.join(user_unpre)

            historical_interactions = ', '.join(historical_interactions)
            gt_list.append(movie_name_dict[testlist[0]])

            if random_n:
                first_name = answer_items_set[0]
                second_name = false_items_set[0]
                tg = 'Yes.'
            else:
                first_name = false_items_set[0]
                second_name = answer_items_set[0]
                tg = 'No.'

                if cm_lable == 'Yes.':
                    cm_lable = 'No.'
                elif cm_lable == 'No.':
                    cm_lable = 'Yes.'

            instruct0 = f'''You are a {kws} recommender system. Based on a user's likes and dislikes, determine if they would prefer one {kws} over another. Respond only with "Yes." or "No.".\n\n'''

            instruct1 = f'''User's Liked {kws}s: <historical interactions>. \nUser's Disliked {kws}s: <user_unpre>\n\nQuestion: Would the user prefer <movie1> over <movie2>?\n'''
            instruct2 = '''Hint: Another recommender system suggests the answer is "<cm_result>"'''
            instruct3 = '\n\nPlease only output "Yes." or "No.".\n'
            if isHint:
                instruct1 = instruct1 + instruct2
            instruct1 = instruct1.replace('<historical interactions>', historical_interactions).replace('<user_unpre>',
                                                                                                        user_unpre).replace(
                '<movie1>', first_name).replace('<movie2>', second_name).replace('<cm_result>', cm_lable)

            fi = {'messages': [
                {"role": "system", "content": [{"type": "text", "content": ""}]},
                {"role": "user", "content": [{"type": "text", "content": instruct0 + instruct1 + instruct3}]},
                {"role": "assistant", "content": [{"type": "text", "content": 'Answer: ' + tg}]}
            ]}
            mes_list.append(fi)
            mes_list_pairwise.append(fi)

            '''
            Listwise Ranking
            '''

           
            print(f" Processing user {uni_int} with candidate items...")
            candidate_movie_ids = get_user_candidate_items_sorted(uni_int, candidate_data)
            
            if not candidate_movie_ids or len(candidate_movie_ids) < 5:
                print(f" User {uni_int} has insufficient candidate items ({len(candidate_movie_ids) if candidate_movie_ids else 0}), skipping listwise...")
                skipped_users += 1
                continue
            
            
            representative_movie_ids, representative_ratings = get_user_representative_items(uni_int, representative_data)
            
            if not representative_movie_ids:
                print(f" No representative items found for user {uni_int}, skipping listwise...")
                skipped_users += 1
                continue
            
            
            user_like_list = representative_movie_ids
            user_dislike_list = []
            
            
            dfp = df_like_p[df_like_p['u'] == uni_int]
            my_list = dfp['i'].tolist()
            my_list_r = dfp['r'].tolist()
            
            for i_ in range(len(my_list)):
                if my_list_r[i_] <= 2:
                    user_dislike_list.append(my_list[i_])
            
            
            if len(user_like_list) > 50:
                user_like_list = user_like_list[:50]
            user_dislike_list = user_dislike_list[:10]

            
            candidate_items = candidate_movie_ids[:5]
            
            
            neglist = []
            while len(neglist) < 5:
                rn = random.sample(movie_id_list, 1)[0]
                if not (rn in candidate_items or rn in neglist):
                    neglist.append(rn)
            
            
            total_list = candidate_items + neglist

            
            total_list_mf = []
            for j_ in total_list:
                try:
                    yy = cm_item[str(j_)]
                    uu = cm_user.get(str(uni_int), None)
                    int_yy = int(j_)
                    
                    int_uu = int(uni_int)
            
                    
                    if int_uu < mf_pred.shape[0] and int_yy < mf_pred.shape[1]:
                    
                        mf_label = mf_pred[int_uu][int_yy]
                        
                    else:
                        mf_label = 1.5
            
                except KeyError:                       
                    mf_label = 1.5
                except IndexError as e:                        
                    mf_label = 1.5
                except ValueError as e:                        
                    mf_label = 1.5
                except Exception as e:                       
                    mf_label = 1.5
            
                total_list_mf.append(mf_label)                       
                                       

            
            total_list_mf_idx = sort_list_reverse_with_indices(total_list_mf)
            total_list_mf_idx = total_list_mf_idx[:5]
            total_list_mf_i = [total_list[k_] for k_ in total_list_mf_idx]

            
            total_list_r = copy.deepcopy(total_list)
            random.shuffle(total_list_r)
            
            
            total_list_t = total_list[:5]

            
            historical_interactions = ', '.join([f'"{movie_name_dict.get(i, f"Movie_{i}")}"' for i in user_like_list])
            neg_interactions = ', '.join([f'"{movie_name_dict.get(i, f"Movie_{i}")}"' for i in user_dislike_list]) if user_dislike_list else 'None.'
            true_answer_items_set = ', '.join([f'"{movie_name_dict[i]}"' for i in total_list_t])
            candidate_item_sets_ = ', '.join([f'"{movie_name_dict[i]}"' for i in total_list_r])
            mf_item_sets_ = ', '.join([f'"{movie_name_dict[i]}"' for i in total_list_mf_i])

           
            historical_triples = get_kg_triples_like_1(user_like_list, kg_less_test_id_df, kg_less_test_df, entity_id_to_name, relation_id_to_description)
            candidate_triples = get_kg_triples_wait_1(total_list, kg_less_test_id_df, kg_less_test_df, entity_id_to_name, relation_id_to_description)
            like_kg = '; '.join(historical_triples) if historical_triples else 'None.'
            wait_kg = '; '.join(candidate_triples) if candidate_triples else 'None.'

            instruct0 = f'''You are a {kws} recommender system. Your task is to rank a given list of candidate {kws}s based on user preferences and return the top five recommendations.\n\n'''

            instruct1 = f'''User's Liked {kws}s: <historical_interactions>. \nUser's Disliked {kws}s: <user_unpre>\n\nQuestion: How would the user rank the candidate item list: <movie_list> based to historical perference?\n'''
            instruct2 = 'Hint: Another recommender model suggests <cm_result>'
            #instruct3 = f'Hint2: These are corresponding entities and relationships for above model\'s recommendation for more context information: <like_kg>, <wait_kg>.\n'
            instruct3 = f'Hint2: These are corresponding entities and relationships for above model’s recommendation for more context information: <like_kg>, <wait_kg>.\n'


            if isHint:
                instruct1 = instruct1 + instruct2

            instruct1 = instruct1.replace('<historical_interactions>', historical_interactions).replace('<user_unpre>', neg_interactions).replace('<movie_list>', candidate_item_sets_).replace('<cm_result>', mf_item_sets_)
            instruct3 = instruct3.replace('<like_kg>', like_kg).replace('<wait_kg>', wait_kg)
            instruct2 = '<|endofmessage|><|assistant|>'
            instruct4 = '\n\nPlease only output the top five recommended movies once in the following format:\n1. [Movie Title]\n2. [Movie Title]\n3. [Movie Title]\n4. [Movie Title]\n5. [Movie Title]\n'

            fi = {'messages': [
                {"role": "system", "content": [{"type": "text", "content": ""}]},
                {"role": "user", "content": [{"type": "text", "content": instruct0 + instruct1 + instruct3 + instruct4}]},
                {"role": "assistant",
                "content": [{"type": "text", "content": 'Answer: ' + str(true_answer_items_set)}]}
            ]}
            mes_list.append(fi)
            mes_list_listwise.append(fi)
                
        else:
            skipped_users += 1
            continue


    #with jsonlines.open(f'./pointwise.jsonl', mode='w') as writer:
        #writer.write_all(mes_list_pointwise)
    #with jsonlines.open(f'./pairwise.jsonl', mode='w') as writer:
        #writer.write_all(mes_list_pairwise)
    print(f" Processing Summary:")
    print(f"  - Total users loaded: {len(sample_list)}")
    print(f"  - Users processed: {processed_users}")
    print(f"  - Users skipped: {skipped_users}")
    print(f"  - Listwise samples generated: {len(mes_list_listwise)}")
    
    with jsonlines.open(f'./listwisetrainlao.jsonl', mode='w') as writer:
        writer.write_all(mes_list_listwise)
    
    print(f" Listwise training data saved to: ./listwisetrain.jsonl")
