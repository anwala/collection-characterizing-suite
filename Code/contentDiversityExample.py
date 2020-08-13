import os, sys

from numpy import linalg as LA
import numpy as np

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

from sklearn.preprocessing import normalize
from sklearn.metrics import pairwise_distances

def genericErrorInfo(slug=''):
    exc_type, exc_obj, exc_tb = sys.exc_info()
    fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
    
    errMsg = fname + ', ' + str(exc_tb.tb_lineno)  + ', ' + str(sys.exc_info())
    print(errMsg + slug)

def getStopwordsSet(frozenSetFlag=False):
    
    stopwords = getStopwordsDict()
    
    if( frozenSetFlag ):
        return frozenset(stopwords.keys())
    else:
        return set(stopwords.keys())

def getStopwordsDict():

    stopwordsDict = {
        "a": True,
        "about": True,
        "above": True,
        "across": True,
        "after": True,
        "afterwards": True,
        "again": True,
        "against": True,
        "all": True,
        "almost": True,
        "alone": True,
        "along": True,
        "already": True,
        "also": True,
        "although": True,
        "always": True,
        "am": True,
        "among": True,
        "amongst": True,
        "amoungst": True,
        "amount": True,
        "an": True,
        "and": True,
        "another": True,
        "any": True,
        "anyhow": True,
        "anyone": True,
        "anything": True,
        "anyway": True,
        "anywhere": True,
        "are": True,
        "around": True,
        "as": True,
        "at": True,
        "back": True,
        "be": True,
        "became": True,
        "because": True,
        "become": True,
        "becomes": True,
        "becoming": True,
        "been": True,
        "before": True,
        "beforehand": True,
        "behind": True,
        "being": True,
        "below": True,
        "beside": True,
        "besides": True,
        "between": True,
        "beyond": True,
        "both": True,
        "but": True,
        "by": True,
        "can": True,
        "can\'t": True,
        "cannot": True,
        "cant": True,
        "co": True,
        "could not": True,
        "could": True,
        "couldn\'t": True,
        "couldnt": True,
        "de": True,
        "describe": True,
        "detail": True,
        "did": True,
        "do": True,
        "does": True,
        "doing": True,
        "done": True,
        "due": True,
        "during": True,
        "e.g": True,
        "e.g.": True,
        "e.g.,": True,
        "each": True,
        "eg": True,
        "either": True,
        "else": True,
        "elsewhere": True,
        "enough": True,
        "etc": True,
        "etc.": True,
        "even though": True,
        "ever": True,
        "every": True,
        "everyone": True,
        "everything": True,
        "everywhere": True,
        "except": True,
        "for": True,
        "former": True,
        "formerly": True,
        "from": True,
        "further": True,
        "get": True,
        "go": True,
        "had": True,
        "has not": True,
        "has": True,
        "hasn\'t": True,
        "hasnt": True,
        "have": True,
        "having": True,
        "he": True,
        "hence": True,
        "her": True,
        "here": True,
        "hereafter": True,
        "hereby": True,
        "herein": True,
        "hereupon": True,
        "hers": True,
        "herself": True,
        "him": True,
        "himself": True,
        "his": True,
        "how": True,
        "however": True,
        "i": True,
        "ie": True,
        "i.e": True,
        "i.e.": True,
        "if": True,
        "in": True,
        "inc": True,
        "inc.": True,
        "indeed": True,
        "into": True,
        "is": True,
        "it": True,
        "its": True,
        "it's": True,
        "itself": True,
        "just": True,
        "keep": True,
        "latter": True,
        "latterly": True,
        "less": True,
        "made": True,
        "make": True,
        "may": True,
        "me": True,
        "meanwhile": True,
        "might": True,
        "mine": True,
        "more": True,
        "moreover": True,
        "most": True,
        "mostly": True,
        "move": True,
        "must": True,
        "my": True,
        "myself": True,
        "namely": True,
        "neither": True,
        "never": True,
        "nevertheless": True,
        "next": True,
        "no": True,
        "nobody": True,
        "none": True,
        "noone": True,
        "nor": True,
        "not": True,
        "nothing": True,
        "now": True,
        "nowhere": True,
        "of": True,
        "off": True,
        "often": True,
        "on": True,
        "once": True,
        "one": True,
        "only": True,
        "onto": True,
        "or": True,
        "other": True,
        "others": True,
        "otherwise": True,
        "our": True,
        "ours": True,
        "ourselves": True,
        "out": True,
        "over": True,
        "own": True,
        "part": True,
        "per": True,
        "perhaps": True,
        "please": True,
        "put": True,
        "rather": True,
        "re": True,
        "same": True,
        "see": True,
        "seem": True,
        "seemed": True,
        "seeming": True,
        "seems": True,
        "several": True,
        "she": True,
        "should": True,
        "show": True,
        "side": True,
        "since": True,
        "sincere": True,
        "so": True,
        "some": True,
        "somehow": True,
        "someone": True,
        "something": True,
        "sometime": True,
        "sometimes": True,
        "somewhere": True,
        "still": True,
        "such": True,
        "take": True,
        "than": True,
        "that": True,
        "the": True,
        "their": True,
        "theirs": True,
        "them": True,
        "themselves": True,
        "then": True,
        "thence": True,
        "there": True,
        "thereafter": True,
        "thereby": True,
        "therefore": True,
        "therein": True,
        "thereupon": True,
        "these": True,
        "they": True,
        "this": True,
        "those": True,
        "though": True,
        "through": True,
        "throughout": True,
        "thru": True,
        "thus": True,
        "to": True,
        "together": True,
        "too": True,
        "toward": True,
        "towards": True,
        "un": True,
        "until": True,
        "upon": True,
        "us": True,
        "very": True,
        "via": True,
        "was": True,
        "we": True,
        "well": True,
        "were": True,
        "what": True,
        "whatever": True,
        "when": True,
        "whence": True,
        "whenever": True,
        "where": True,
        "whereafter": True,
        "whereas": True,
        "whereby": True,
        "wherein": True,
        "whereupon": True,
        "wherever": True,
        "whether": True,
        "which": True,
        "while": True,
        "whither": True,
        "who": True,
        "whoever": True,
        "whole": True,
        "whom": True,
        "whose": True,
        "why": True,
        "will": True,
        "with": True,
        "within": True,
        "without": True,
        "would": True,
        "yet": True,
        "you": True,
        "your": True,
        "yours": True,
        "yourself": True,
        "yourselves": True
    }
    
    return stopwordsDict

def add0sToMainDiag(squareMat):        
    row, col = squareMat.shape
    if( row != col ):
        return 
    for i in range(row):
        squareMat[i][i] = 0

def getSimOrDistMatrix(matrix, matrixType='sim'):
        
    matrix = np.array(matrix)
    matrix = pairwise_distances(matrix, metric='cosine')

    if( matrixType == 'sim' ):
        matrix = 1 - matrix

    return matrix.tolist()

def getTFMatrixFromDocList(oldDocList, params=None):
    
    if( len(oldDocList) == 0 ):
        return []

    docList = []
    #remove empty documents
    for i in range(len(oldDocList)):
        if( len(oldDocList[i]) != 0 ):
            docList.append( oldDocList[i] )

    if( len(docList) == 0 ):
        return []

    if( params is None ):
        params = {}

    
    params.setdefault('idf', {'active': False, 'norm': None})#see TfidfTransformer for options

    params.setdefault('normalize', False)#normalize TF by vector norm (L2 norm)
    params.setdefault('ngram_range', (1, 1))#normalize TF by vector norm (L2 norm)
    params.setdefault('tokenizer', None)
    params.setdefault('token_pattern', r'(?u)\b[a-zA-Z\'\’-]+[a-zA-Z]+\b|\d+[.,]?\d*')
    params.setdefault('custom_vocab', None)
    params.setdefault('verbose', False)
    params.setdefault('extra_payload', False)

    if( params['custom_vocab'] is None ):
        stopwords = getStopwordsSet()
    else:
        stopwords = None

    countVectorizer = CountVectorizer(tokenizer=params['tokenizer'], token_pattern=params['token_pattern'], vocabulary=params['custom_vocab'], stop_words=stopwords, ngram_range=params['ngram_range'])
    termFreqMatrix = countVectorizer.fit_transform(docList)

    if( params['normalize'] ):
        termFreqMatrix = normalize(termFreqMatrix, norm='l2', axis=1)

    if( params['idf']['active'] ):
        tfidf = TfidfTransformer( norm=params['idf']['norm'] )
        tfidf.fit(termFreqMatrix)

        tfIdfMatrix = tfidf.transform(termFreqMatrix)
        dense = tfIdfMatrix.todense()
    else:
        dense = termFreqMatrix.todense()
    
        

    dense = dense.tolist()
    if( params['extra_payload'] ):
        
        vocabDict = {}
        for vocab, pos in countVectorizer.vocabulary_.items():
            vocabDict.setdefault(
                vocab, {
                    'tf': 0,
                    'indx': int(pos)
                }
            )

            for row in dense:
                vocabDict[vocab]['tf'] += row[pos]

        return {
            'tf_matrix': dense,
            'feature_matrix': countVectorizer.get_feature_names(),
            'sorted_vocab': sorted(vocabDict.items(), key=lambda x: x[1]['tf'], reverse=True)
        }
    else:
        return dense

def getColSimScore(normalizedTFIDFMatrix, simMatInput=False):
        
    if( len(normalizedTFIDFMatrix) == 0 ):
        return -1

    similarityScore = -1

    try:
        if( simMatInput is True ):
            simMatrix = normalizedTFIDFMatrix
        else:
            simMatrix = getSimOrDistMatrix(normalizedTFIDFMatrix)
        
        docMat = np.array(simMatrix)

        row, col = docMat.shape
        if( row != col or row < 2 ):
            return -1

        onesMat = np.ones((row, row))        

        add0sToMainDiag( docMat )
        add0sToMainDiag( onesMat )

        docMatNorm = LA.norm( docMat )
        onesMatNorm = LA.norm( onesMat )

        similarityScore = docMatNorm/onesMatNorm
    except:
        genericErrorInfo()

    return similarityScore

def jaccardFor2Sets(firstSet, secondSet):

    intersection = float(len(firstSet & secondSet))
    union = len(firstSet | secondSet)

    if( union != 0 ):
        return  round(intersection/union, 4)
    else:
        return 0

def overlapFor2Sets(firstSet, secondSet):

    intersection = float(len(firstSet & secondSet))
    minimum = min(len(firstSet), len(secondSet))

    if( minimum != 0 ):
        return  round(intersection/minimum, 4)
    else:
        return 0

def weightedJaccardOverlapSim(firstSet, secondSet, jaccardWeight=0.4, overlapWeight=0.6):
        
    if( jaccardWeight + overlapWeight != 1 ):
        return -1
    
    return (jaccardWeight * jaccardFor2Sets(firstSet, secondSet)) + (overlapWeight * overlapFor2Sets(firstSet, secondSet))

def getColEntitySimScore(entyLinks, params=None):

    if( params is None ):
        params = {}

    if( 'sim-coeff' not in params ):
        #params['sim-coeff'] = 0.3
        params['sim-coeff'] = 0.27

    if( 'jaccard-weight' not in params ):
        params['jaccard-weight'] = 0.4

    if( 'overlap-weight' not in params ):
        params['overlap-weight'] = 0.6


    #consider optimizing this because sim matrix is symmetrical j,i can look up i,j
    simMatrix = []

    for i in range(0, len(entyLinks)):
        simMatrix.append( [-1]*len(entyLinks) )
        for j in range(0, len(entyLinks)):

            sim = 0
            if( i == j ):
                sim = 1
            else:
                sim = weightedJaccardOverlapSim(
                    entyLinks[i], 
                    entyLinks[j],
                    params['jaccard-weight'],
                    params['overlap-weight']
                )
                
                if( sim >= params['sim-coeff'] ):
                    sim = 1
                else:
                    sim = 0

            simMatrix[i][j] = sim

    return getColSimScore(simMatrix, True)

docList = ['Julie loves me more than Linda loves me',
'Jane likes me more than Julie loves me',
'He likes basketball more than baseball']

TFIDFMatrix = getTFMatrixFromDocList( docList )
cosineDiv = getColSimScore( TFIDFMatrix )
if( cosineDiv == -1 ):
    print('ERROR, cosineDiv = -1')
    cosineDiv = 0
else:
    cosineDiv = 1 - cosineDiv
print('cosineDiv (Document-Term Matrix representation):', cosineDiv)

#entities can be extracted with Stanford CoreNLP:
#https://stanfordnlp.github.io/stanza/
#https://ws-dl.blogspot.com/2018/03/2018-03-04-installing-stanford-corenlp.html
#https://stanfordnlp.github.io/CoreNLP/other-languages.html#docker
entityCol = [
    set(['julie', 'linda']),
    set(['jane', 'julie']),
    set([])
]
entityDiversity = getColEntitySimScore( entityCol )   
if( entityDiversity == -1 ):
    print('ERROR, entityDiversity = -1')
    entityDiversity = 0
else:
    entityDiversity = 1 - entityDiversity
print('entityDiversity (Entity-Set representation):', entityDiversity)
