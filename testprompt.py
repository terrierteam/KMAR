import pandas as pd
import random
import pickle
import jsonlines

random.seed(42)

dataset_names = ['ml-1m']
model_names = ['LightGCN']
isHint = True

#load kg info 
output_df = pd.read_csv('output22.csv', sep='|', header=None, names=['movie_id', 'movie_name', 'encoding'])
entity_df = pd.read_csv('only_entity-id.tsv', sep='\t', header=None, names=['encoding', 'translate1', 'translate2', 'translate3', 'entity_id'])
kg_less_test_df = pd.read_csv('pretrain-_text_3.tsv', sep='\t', header=None, names=['head', 'relation', 'tail'])
kg_less_test_id_df = pd.read_csv('pretrain-_kg_id_3.tsv', sep='\t', header=None, names=['head_id', 'relation_id', 'tail_id'])

relations_df = pd.read_csv('all_relations_id.tsv', sep='\t', header=None, names=['relation_id', 'relation'])
movie_name_dict = dict(zip(output_df['movie_id'], output_df['movie_name']))
movie_id_to_encoding = dict(zip(output_df['movie_id'], output_df['encoding']))
entity_id_to_name = dict(zip(entity_df['entity_id'], entity_df['translate1']))
relation_id_to_description = dict(zip(relations_df['relation_id'], relations_df['relation']))

#input kg_triples
def get_kg_triples_like_random(book_list, kg_id_df, kg_text_df, entity_to_name, relation_to_desc):
    triples = []
    
    for book in book_list:
        related_triples = kg_id_df[(kg_id_df['head_id'] == book) | (kg_id_df['tail_id'] == book)]
        
        if related_triples.empty:
        else:
            selected_triples = related_triples.sample(n=min(1, len(related_triples)), random_state=42)
            
            for _, triple in selected_triples.iterrows():
                head_text = kg_text_df.iloc[triple.name]['head']
                relation_text = kg_text_df.iloc[triple.name]['relation']
                tail_text = kg_text_df.iloc[triple.name]['tail']
                
                triples.append(f'{head_text} - {relation_text} - {tail_text}')
    
    return triples

def sort_list_reverse_with_indices(lst):
    sorted_indices = sorted(enumerate(lst), key=lambda x: x[1], reverse=True)
    sorted_indices = [index for index, _ in sorted_indices]
    return sorted_indices

for dataset_name in dataset_names:

    with open('./ml-1m_item.pkl', "rb") as file:
        cm_item = pickle.load(file)
    with open('./ml-1m_user.pkl', "rb") as file:
        cm_user = pickle.load(file)
    with open('./ml-1m_pred.pkl', "rb") as file:
        cm_pred = pickle.load(file)
    with open('./ml-1m_item_id_mapping.pkl', "rb") as file:
        mf_item = pickle.load(file)
    with open('./ml-1m_user_id_mapping.pkl', "rb") as file:
        mf_user = pickle.load(file)
    with open('./ml-1m_rating_matrix.pkl', "rb") as file:
        mf_pred = pickle.load(file)

    kws = 'movie' if 'book' not in dataset_name else 'book'
    kk = 5

    for model_name in model_names:
        rec_list = pd.read_csv('./LightGCNrec_save_dict1.csv', header=None,
                               names=['v' + str(i) for i in range(11)])
        gt_list = pd.read_csv('./LightGCNgt_save_dict1.csv', header=None,
                              names=['u', 'i'])

        if dataset_name == 'ml-1m':
            df_like = pd.read_csv('./train_set.txt', names=['u', 'i', 'r', 't'], sep=' ')
            movie_info = pd.read_csv('./movie_info_ml1m.csv', header=None,names=['movie_id', 'movie_name', 'year', 'genre'], sep='|', engine='python', encoding='latin-1')
            
            movie_id_list = movie_info['movie_id'].tolist()
            movie_name_list = movie_info['movie_name'].tolist()
            movie_name_dict.update({movie_id_list[i]: movie_name_list[i] for i in range(len(movie_name_list))})

        mes_list_listwise = []

        for idx, row in rec_list.iterrows():
            uni = row['v0']

            df = df_like[df_like['u'] == uni]
            my_list = df['i'].tolist()
            random.shuffle(my_list)
            my_list = [int(x) for x in my_list]

            if len(df) > 55:
                topk = 50
            else:
                topk = max(5, len(df) - 3)
            trainlist = my_list[:topk]

            testlist_ = [row['v' + str(ii)] for ii in range(1, 1 + 2 * kk)]
            
            #input historical_triples
            historical_triples = get_kg_triples_like_random(trainlist, kg_less_test_id_df, kg_less_test_df, entity_id_to_name, relation_id_to_description)
            #input candidate_triples
            candidate_triples = get_kg_triples_like_random(testlist_, kg_less_test_id_df, kg_less_test_df, entity_id_to_name, relation_id_to_description)

            like_kg = '; '.join(historical_triples) if historical_triples else 'None.'
            wait_kg = '; '.join(candidate_triples) if candidate_triples else 'None.'

            candidate_item_sets_ = ', '.join([f'"{movie_name_dict[i]}"' for i in testlist_])

            total_list_mf = []
            for j_ in testlist_:
                try:
                    yy = cm_item[str(j_)]
                    uu = cm_user.get(str(uni), None)
                    int_yy = int(j_)
                    int_uu = int(uni)
                    if int_uu < mf_pred.shape[0] and int_yy < mf_pred.shape[1]:
                        mf_label = mf_pred[int_uu][int_yy]
                    else:
                        mf_label = 1.5
                except Exception:
                    mf_label = 1.5
                total_list_mf.append(mf_label)

            total_list_mf_idx = sort_list_reverse_with_indices(total_list_mf)
            total_list_mf_idx = total_list_mf_idx[:5]
            total_list_mf_i = [testlist_[k_] for k_ in total_list_mf_idx]
            mf_item_sets_ = ', '.join([f'"{movie_name_dict[i]}"' for i in total_list_mf_i])
            candidate_item_sets_ = ', '.join([f'"{movie_name_dict[i]}"' for i in testlist_])

            instruct0 = f'''You are a {kws} recommender system. Your task is to rank a given list of candidate {kws}s based on user preferences and return the top five recommendations.\n\n'''

            instruct1 = f'''User's Liked {kws}s: <historical_interactions>. \nUser's Disliked {kws}s: None\n\nQuestion: How would the user rank the candidate item list: <movie_list> based on historical preference?\n'''
            instruct2 = 'Hint: Another recommender model suggests <cm_result>'

            instruct3 = f'Hint2: These are corresponding entities and relationships for above model’s recommendation for more context information: <like_kg>, <wait_kg>.\n'
            instruct1 = instruct1.replace('<historical_interactions>', ', '.join([f'"{movie_name_dict[i]}"' for i in trainlist])).replace('<movie_list>', candidate_item_sets_).replace('<cm_result>', mf_item_sets_)
            instruct3 = instruct3.replace('<like_kg>', like_kg).replace('<wait_kg>', wait_kg)

            instruct2 = '<|endofmessage|><|assistant|>'
            instruct4 = '\n\nPlease only output the top five recommended movies once in the following format:\n1. [Movie Title]\n2. [Movie Title]\n3. [Movie Title]\n4. [Movie Title]\n5. [Movie Title].\n'
            fi = {'inst': instruct0 + instruct1 + instruct3 + instruct4 + instruct2}
            mes_list_listwise.append(fi)

        with jsonlines.open(f'./test.jsonl', mode='w') as writer:
            writer.write_all(mes_list_listwise)

        print(f"Listwise ranking task generation completed for model: ml-1m_lightKG")
