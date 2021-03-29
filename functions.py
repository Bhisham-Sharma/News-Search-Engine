import nltk
nltk.download('stopwords')
nltk.download('framenet_v17')
from nltk.stem import SnowballStemmer
import re
from nltk.corpus import stopwords


stop_words = stopwords.words('english')
###### ALL Required Functions ######

# preprocessing
def preprocess_words(query):
    list_patterns = [r'\d+','<.*?>','[^a-zA-Z]',' +',r'^https?:\/\/.*[\r\n]*',r'\b\w{1,3}\b']    # list of patterns
    reg_expression = [re.compile(p) for p in list_patterns]

    sno = SnowballStemmer('english')
    
    r = re.sub(reg_expression[4]," ",str(query)) # remove url
    r = re.sub(reg_expression[1]," ",r)     # remove html tags
    r = re.sub(reg_expression[0],"",r)  # remove sequence of digit
    r = re.sub(reg_expression[2]," ",r) # remove everything except alphabets.
    r = re.sub(reg_expression[5],"",r)     # remove words less than three characters
    r = re.sub(reg_expression[3]," ",r) # replace two consecutive space into one
    final = r.lower()                        # all words to lowercase
    final = ' '.join([word for word in final.split()\
                            if word not in stop_words])      #removing stopwords
    final = ' '.join([sno.stem(word) for word in final.split()])  #Snowball stemming
    
    return final


# return clean suggestions
def clean_suggestions(query):
    list_patterns = [r'\d+','<.*?>','[^a-zA-Z]',' +',r'^https?:\/\/.*[\r\n]*',r'\b\w{1,3}\b']    # list of patterns
    reg_expression = [re.compile(p) for p in list_patterns]
    
    r = re.sub(reg_expression[4]," ",str(query)) # remove url
    r = re.sub(reg_expression[1]," ",r)     # remove html tags
    r = re.sub(reg_expression[0],"",r)  # remove sequence of digit
    r = re.sub(reg_expression[2]," ",r) # remove everything except alphabets.
    r = re.sub(reg_expression[5],"",r)     # remove words less than three characters
    r = re.sub(reg_expression[3]," ",r) # replace two consecutive space into one
    final = r.lower()                        # all words to lowercase
    
    return final


# clean data without stemming
def clean_data(query):
    list_patterns = [r'\d+','<.*?>','[^a-zA-Z]',' +',r'^https?:\/\/.*[\r\n]*',r'\b\w{1,3}\b']    # list of patterns
    reg_expression = [re.compile(p) for p in list_patterns]
    
    r = re.sub(reg_expression[4]," ",str(query)) # remove url
    r = re.sub(reg_expression[1]," ",r)     # remove html tags
    r = re.sub(reg_expression[2]," ",r) # remove everything except alphabets.
    r = re.sub(reg_expression[3]," ",r) # replace two consecutive space into one
    
    return r

# get releated words to each word: Query expansion
def related_words(word):
    from nltk.corpus import framenet as fn
    return [f.name for f in fn.frames_by_lemma(word)]


# linear merge function with position difference 5
# the query to be passed here must be pre processed
# add words similar to the query words
def linearMergePosition(query, index_dict):
    query = ' '.join([word for word in query.lower().split() if word not in stop_words])
    cleaned_query = ""
    
    if len(query.split(" ")) < 2:
        full_query = ""
        for word in query.split(" "):
            full_query += word +" "+" ".join(related_words(word))+ " "
        cleaned_query = preprocess_words(full_query)
    else:
        cleaned_query = preprocess_words(query)
    
    index_of_words = []
    for word in cleaned_query.split(" "):
        if word in index_dict:
            index_of_words.append(index_dict.get(word))
        else:
            pass
    
    first_index = []
    for i in range(0,len(cleaned_query.split(" "))):
        try:
            first_index.append(index_of_words[i][0][0])
        except:
            pass

    min_index = first_index.index(min(first_index))
    index_of_words[min_index], index_of_words[0] = index_of_words[0], index_of_words[min_index]
    
    found_docs = set()
    all_docs = set()
    for i in range(0,len(index_of_words)):
        for j in range(0,len(index_of_words[i])):
            compare = index_of_words[i][j][0]
            all_docs.add(compare)
            
            for k in range(i+1,len(index_of_words)):
                for m in range(0,len(index_of_words[k])):
                    if compare == index_of_words[k][m][0]:
                        if index_of_words[i][j][1] - index_of_words[k][m][1] <= 5:
                            found_docs.add(index_of_words[i][j][0])
    
    if len(found_docs) == 0:
        return all_docs
    else:
        return found_docs


# spell checker
def spellCheck(query):
    from spellchecker import SpellChecker
    spell = SpellChecker()
    misspelled = spell.unknown(query.split(" "))
    if len(misspelled) == 0:
        return query
    else:
        for word in misspelled:
            query = query.replace(word, spell.correction(word))
        return query