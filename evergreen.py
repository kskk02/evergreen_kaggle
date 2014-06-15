import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
import ast

def convert_text (df, col):
    title_list = []
    body_list = []
    url_list = []
    for (i, row) in enumerate(df[col]):
        try:
            bp_dict = ast.literal_eval(row)
        except:
            bp_dict = {}
        for k,v in bp_dict.iteritems():
            if k == 'title':
                title_list.append(v)
            if k == 'body':
                body_list.append(v)
            if k == 'url':
                url_list.append(v)
        if len(title_list) == i:
            title_list.append('')
        if len(body_list) == i:
            body_list.append('')
        if len(url_list) == i:
            url_list.append('') 
    return title_list, body_list, url_list 
    
def nums_in_title (df, title_col):
    title_words = [s.split() for s in df[title_col]]
    num_list = []
    for title in title_words:
        title_nums = [int(word) for word in title if word.isdigit()]
        num_list.append(any([n for n in title_nums]))
    return num_list
    
def find_tags (df, body_col):
    tag_list = []
    has_tags = []
    for body in df[body_col]:
        spaces = 0
        tags = []
        for index, ch in enumerate(reversed(body)):
            if ch == ',':
                spaces = 0
            if ch == ' ':
                spaces += 1
            if spaces > 4:
                if tags != []:
                    
                

if __name__ == "__main__":
    su_df = pd.read_table('train.tsv')
    
    su_df['title'], su_df['body'], su_df['url_text'] = convert_text(su_df, 'boilerplate')
    
    su_df['nums'] = nums_in_title(su_df, 'title')
    
    
    
    
    
        
                
    
    
    
# "url"    "urlid"    "boilerplate"    "alchemy_category"    "alchemy_category_score"    
# "avglinksize"    "commonlinkratio_1"    "commonlinkratio_2"    "commonlinkratio_3"
# "commonlinkratio_4"    "compression_ratio"    "embed_ratio"    "framebased"    "frameTagRatio"    
# "hasDomainLink"    "html_ratio"    "image_ratio"    "is_news"    "lengthyLinkDomain"    
# "linkwordscore"    "news_front_page"    "non_markup_alphanum_characters"    "numberOfLinks"    
# "numwords_in_url"    "parametrizedLinkRatio"    "spelling_errors_ratio"    "label"

    
    

