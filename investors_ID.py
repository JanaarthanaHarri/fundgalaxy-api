import pandas as pd
import scipy.sparse as sp
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def get_data():
    inv_data = pd.read_csv('datasets/investors_data.csv')
    # inv_data['name'] = inv_data['name'].str.lower()
    return inv_data

def combine_data(data):
    drop_cols = ['uuid', 'name', 'partners', 'tot_funding', 'address',
       'linkedin', 'domain', 'description', 'invested_companies']
    data_recommend = data.drop(columns=drop_cols)
    data_recommend['combine'] = data_recommend[data_recommend.columns[0:2]].apply(
                                     lambda x: ','.join(x.dropna().astype(str)),axis=1)
        
    data_recommend = data_recommend.drop(columns=[ 'type', 'investment_type'])
    return data_recommend

def transform_data(data_combine, data_descr):
        count = CountVectorizer(stop_words='english')
        count_matrix = count.fit_transform(data_combine['combine'])

        tfidf = TfidfVectorizer(stop_words='english')
        tfidf_matrix = tfidf.fit_transform(data_descr['description'])

        combine_sparse = sp.hstack([count_matrix, tfidf_matrix], format='csr')
        cosine_sim = cosine_similarity(combine_sparse, combine_sparse)
        
        return cosine_sim

def recommend_investors(investor_id, data, combine, transform):
        indices = pd.Series(data.index, index = data['uuid'])
        index = indices[investor_id]

        sim_scores = list(enumerate(transform[index]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        sim_scores = sim_scores[0:5]

        investor_indices = [i[0] for i in sim_scores]

        uuid = data['uuid'].iloc[investor_indices]
        name = data['name'].iloc[investor_indices]
        description = data['description'].iloc[investor_indices]
        domain = data['domain'].iloc[investor_indices]
        total_funding = data['tot_funding'].iloc[investor_indices]
        address = data['address'].iloc[investor_indices]
        partners = data['partners'].iloc[investor_indices]
        linkedin_url = data['linkedin'].iloc[investor_indices]
        invested_companies = data['invested_companies'].iloc[investor_indices]
        investor_type = data['type'].iloc[investor_indices]
        investment_type = data['investment_type'].iloc[investor_indices]

        recommend_cols = ['ID','Name', 'Description','Domain','Partners','Type','Address','Total Funding',
                           'Invested Companies','Investment Type','Linkedin']
        recommendation_data = pd.DataFrame(columns=recommend_cols)

        recommendation_data['ID'] = uuid
        recommendation_data['Name'] = name
        recommendation_data['Description'] = description
        recommendation_data['Domain'] = domain
        recommendation_data['Partners'] = partners
        recommendation_data['Total Funding'] = total_funding
        recommendation_data['Invested Companies'] = invested_companies
        recommendation_data['Type'] = investor_type
        recommendation_data['Address'] = address
        recommendation_data['Linkedin'] = linkedin_url
        recommendation_data['Investment Type'] = investment_type

        return recommendation_data

def results(investor_id):
        # investor_name = investor_name.lower()

        find_investor = get_data()
        combine_result = combine_data(find_investor)
        transform_result = transform_data(combine_result,find_investor)

        if investor_id not in find_investor['uuid'].unique():
                return 'investor not in Database'

        else:
                recommendations = recommend_investors(investor_id, find_investor, combine_result, transform_result)
                return recommendations.to_dict('records')