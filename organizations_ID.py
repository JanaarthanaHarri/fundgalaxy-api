import pandas as pd
import scipy.sparse as sp
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def get_data():
    org_data = pd.read_csv('datasets/organizations_data.csv')
    # org_data['name'] = org_data['name'].str.lower()
    return org_data

def combine_data(data):
    drop_cols = ['uuid', 'name', 'description', 'domain', 'total_funding',
       'num_funding_rounds', 'total_funding_currency_code', 'founded_on',
       'employee_count', 'email', 'phone','address','facebook_url', 'linkedin_url',
       'twitter_url', 'people', 'founder']
    data_recommend = data.drop(columns=drop_cols)
    data_recommend['combine'] = data_recommend[data_recommend.columns[0:2]].apply(
                                     lambda x: ','.join(x.dropna().astype(str)),axis=1)
        
    data_recommend = data_recommend.drop(columns=[ 'category', 'category_grp_list'])
    return data_recommend

def transform_data(data_combine, data_descr):
        count = CountVectorizer(stop_words='english')
        count_matrix = count.fit_transform(data_combine['combine'])

        tfidf = TfidfVectorizer(stop_words='english')
        tfidf_matrix = tfidf.fit_transform(data_descr['description'])

        combine_sparse = sp.hstack([count_matrix, tfidf_matrix], format='csr')
        cosine_sim = cosine_similarity(combine_sparse, combine_sparse)
        
        return cosine_sim

def recommend_companies(company_id, data, combine, transform):
        indices = pd.Series(data.index, index = data['uuid'])
        index = indices[company_id]

        sim_scores = list(enumerate(transform[index]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        sim_scores = sim_scores[0:5]

        company_indices = [i[0] for i in sim_scores]

        uuid = data['uuid'].iloc[company_indices]
        name = data['name'].iloc[company_indices]
        description = data['description'].iloc[company_indices]
        domain = data['domain'].iloc[company_indices]
        total_funding = data['total_funding'].iloc[company_indices]
        num_funding_rounds = data['num_funding_rounds'].iloc[company_indices]
        total_funding_currency_code = data['total_funding_currency_code'].iloc[company_indices]
        founded_on = data['founded_on'].iloc[company_indices]
        employee_count = data['employee_count'].iloc[company_indices]
        phone = data['phone'].iloc[company_indices]
        address = data['address'].iloc[company_indices]
        facebook_url = data['facebook_url'].iloc[company_indices]
        linkedin_url = data['linkedin_url'].iloc[company_indices]
        twitter_url = data['twitter_url'].iloc[company_indices]
        people = data['people'].iloc[company_indices]
        founder = data['founder'].iloc[company_indices]
        category = data['category'].iloc[company_indices]

        recommend_cols = ['ID','Name', 'Description','Domain','Founder','People','Category','Total Funding',
                           'Num Funding Rounds','Funding Currency Code','Founded On','Employee Count','Phone',
                           'Address','Facebook','Linkedin','Twitter']
        recommendation_data = pd.DataFrame(columns=recommend_cols)

        recommendation_data['ID'] = uuid
        recommendation_data['Name'] = name
        recommendation_data['Description'] = description
        recommendation_data['Domain'] = domain
        recommendation_data['Founder'] =  founder
        recommendation_data['People'] = people
        recommendation_data['Category'] = category
        recommendation_data['Total Funding'] = total_funding
        recommendation_data['Num Funding Rounds'] = num_funding_rounds
        recommendation_data['Funding Currency Code'] = total_funding_currency_code
        recommendation_data['Founded On'] = founded_on
        recommendation_data['Employee Count'] = employee_count
        recommendation_data['Phone'] = phone
        recommendation_data['Address'] = address
        recommendation_data['Facebook'] = facebook_url
        recommendation_data['Linkedin'] = linkedin_url
        recommendation_data['Twitter'] = twitter_url

        return recommendation_data

def results(company_id):
        # company_id = company_id.lower()

        find_company = get_data()
        combine_result = combine_data(find_company)
        transform_result = transform_data(combine_result,find_company)

        if company_id not in find_company['uuid'].unique():
                return 'Company not in Database'

        else:
                recommendations = recommend_companies(company_id, find_company, combine_result, transform_result)
                return recommendations.to_dict('records')