import argparse


parser = argparse.ArgumentParser(description='Knowledge-based NLP model for re')
parser.add_argument('-i', help='Job identity', type=int, default=0)
parser.add_argument('-rb', help='Rebuild data', type=int, default=0)
parser.add_argument('-e', help='Number of epochs', type=int, default=80)
parser.add_argument('-p', help='Patience of early stop (0 to ignore)', type=int, default=10)
parser.add_argument('-len', help='Max sentence or document length', type=int, default=100)
parser.add_argument('-d', help='dropout rate', type=int, default=0.5)
parser.add_argument('-config', help='CNN configurations default \'1:128\'', type=str, default='1:32,2:96,3:32,4:64')  # 2:32,3:96,4:32,5:64
parser.add_argument('-sc', help='scoring function for TransE', type=str, default='l2')

opt = parser.parse_args()
print('Running opt: {}'.format(opt))

JOB_IDENTITY = opt.i
IS_REBUILD = opt.rb
EPOCHS = opt.e
EARLY_STOPPING = False if opt.p == 0 else True
MAX_LENGTH = opt.len
DROPOUT = opt.d
PATIENCE = opt.p
SCORE = opt.sc

CNN_FILTERS = {}
if opt.config:
    print('Use model CNN with config', opt.config)
    USE_CNN = True
    CNN_FILTERS = {
        int(k): int(f) for k, f in [i.split(':') for i in opt.config.split(',')]
    }
else:
    raise ValueError('Configure CNN model to start')

UNK = '$UNK$'

ALL_LABELS = ['CID', 'NONE']


columns = {'ChemicalName', 'ChemicalID', 'CasRN', 'DiseaseName', 'DiseaseID', 'DirectEvidence', 'InferenceGeneSymbol',
           'InferenceScore', 'OmimIDs', 'PubMedIDs'}

list_relations = ['inferred-association', 'therapeutic', 'marker/mechanism', 'other']

DATA = 'data/'

CTD = DATA + 'ctd/CTD_chemicals_diseases.csv'

CDR = DATA + 'cdr/cdr_'

PICKLE = DATA + 'pickle/'

cdr_datasets = ['train.txt', 'dev.txt', 'test.txt']

ENTITY_PATH = DATA + 'entity_representation/'

WORDNET_PATH = DATA + 'wordnet/'

INPUT_W2V_DIM = 768

TRAINED_MODELS = DATA + 'trained_models/'
MODEL_NAMES = TRAINED_MODELS + '{}_{}'

SDP = DATA + 'sdp/'
W2V_DATA = DATA + 'w2v_model/'

ALL_WORDS = DATA + 'all_words.txt'
ALL_POSES = DATA + 'all_poses.txt'
ALL_SYNSETS = DATA + 'all_hypernyms.txt'
# ALL_DEPENDS = DATA + 'all_relations.txt'
ALL_DEPENDS = DATA + 'all_depend.txt'


TRIMMED_W2V = W2V_DATA + 'biowordvec_nlplab.npz'

# Sentence types
SENTENCE_TYPE_GENERAL = 0
SENTENCE_TYPE_TITLE = 1
SENTENCE_TYPE_ABSTRACT = 2

# Entity types
ENTITY_TYPE_GENERAL = 0
ENTITY_TYPE_DISEASE = 1
ENTITY_TYPE_CHEMICAL = 2

# BioEntity id keys
MESH_KEY = 'mesh_id'
BRAT_KEY = 'brat_id'

# Relation types
RELATION_TYPE_GENERAL = 0
RELATION_TYPE_CID = 1
RELATION_TYPE_TREAT = 2

VERB_TAGS = {'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ'}
PRONOUN_TAGS = {'PRP'}

# string that mention to a chemical
CHEMICAL_MENTIONS = {
    'chemical', 'chemicals', 'drug', 'drugs', 'medicine', 'medicines', 'therapy', 'therapies', 'cure', 'cures',
    'medication', 'medications', 'medicament', 'curative', 'curatives', 'homeopathy', 'revulsion', 'usage', 'usages',
    'physics', 'treatment', 'treatments', 'sulphite', 'sulphites', 'dose', 'doses', 'dosage', 'dosages '
}
# string that mention to a disease
DISEASE_MENTIONS = {
    'disease', 'diseases', 'illness', 'illnesses', 'pain', 'pains', 'symptom', 'symptoms', 'syndrome', 'syndromes',
    'case', 'cases', 'sign', 'signs', 'patient', 'patients', 'phenomena', 'phenomenas', 'phenomenon', 'phenomenons',
    'phenotype', 'phenotypes', 'prognosis', 'prognoses', 'foretoken', 'foretokens'
}

STOP_WORD = {
    'been', 'than', 'can', 'selves', 'index', 'becomes', 'date', 'inner', 'hereby', 'looks', 'that', 'non', 't',
    'giving', 'unlikely', 'help', 'definitely', 'do', 'latter', 'at ', "haven't", 'namely', 'fix', 'therein', 'eight',
    'hi', 'along', 'seriously', 'keep    keeps', 'beginnings', 'also', 'welcome', 'importance', 'nonetheless', 'vs',
    'un', 'went', 'until', 'truly', 'whos', 'take', 'liked', 'am', 'nothing', 'now', 'anyone', 'all', 'sup', 'inward',
    'currently', 'an ', 'where', 'noted', 'f', 'mainly', 'next', 'given', 'mostly', 'act', "'s", 'somewhere', "she'll",
    'allow', 'got', 'anyhow', 'begin', 'wed', 'it', "wasn't", 'et', 'everyone', "'ve", 'oh', 'viz', 'on', 'found',
    "t's", 'thousand', "didn't", 'forth', 'despite', 'accordingly', 'anyway', 'as', 'is', "doesn't", 'results', 'at',
    'example', 'necessary', 'particular', 'n', "who's", "he's", 'beginning', 'suggest', 'most', 'affecting', 'goes',
    'followed', 'was ', 'anything', 've', 'elsewhere', 'nowhere', 'not', 'sensible', 'h', 'pp', 'o', 'itself', 'okay',
    'w', 'seem', 'obviously', 'whom', 'there', 'immediate', 'g', "that've", 'ran', 'recent', 'mg', 'last', 'above',
    'ending', 'nine', 'ask', 'important', 'whats', 'therere', 'whence', 'adj', 'werent', "ain't", 'away', 'over',
    'presumably', 'whod', 'hid', 'already', 'please', 'third', 'hed', 'moreover', 'tried', 'gone', 'ignored', 'who ',
    'better', 'wherein', 'consider', 'keeps', 'first', 'howbeit', 'throughout', 'associated', 'herein', 'give', 'y',
    'serious', 'else', 'sha', 'other', 'perhaps', 'everything', 'rather', "you've", 'ought', 'look', 'doing', 'knows',
    'exactly', 'have', 'ai', 'on ', 'causes', 'before', 'right', 'want', 'somehow', 'ever', 'her', "he'd", 'or ', 'ltd',
    'be ', 'of', 'within', "wouldn't", 'merely', 'ourselves', 'r', "weren't", 'just', 'allows', 'twice', 'three',
    "here's", 'aside', 'they', 'what', 'specifically', 'you', 'refs', 'when', 'later', 'since', 'recently', 's',
    'means', 'whatever', 'became', 'self', 'normally', "there's", 'mr', 'mean', 'almost', 'had', 'only', 'very',
    "hadn't", "c's", 'them', 'toward', 'thats', 'successfully', 'beside', 'under', 'shes', 'around', 'try', 'why',
    'came', 'section', 'respectively', 'gotten', 'same', 'six', 'below', 'j', 'appreciate', 'c', 'nevertheless', "a's",
    'ok', 'whose', 'well', 'pages', 'far', 'cannot', 'nos', 'changes', 'likely', 'may', 'consequently', "it'd",
    'reasonably', 'million', 'someone', 'onto', 'theyre', 'whenever', 'he', 'if', 'were', 'run', 'appropriate', 'say',
    "shouldn't", 'regards', 'miss', 'clearly', 'heres', 'myself', 'ts', 'substantially', 'thereto', 'let', 'et-al',
    'wheres', 'thereafter', 'after', 'per', 'provides', 'each', 'keep', 'anyways', "why's", 'or', 'briefly', "how's",
    'get', "'m", "it'll", 'promptly', 'once', 'seems', 'wonder', 'uses', 'ff', 'various', "c'mon", 'nobody', 'na',
    'again', 'indeed', 'therefore', 'saying', 'alone', 'two', 'm', 'but', 'l', 'will ', "what's", 'past', "there'll",
    "i'd", 'described', 'ed', 'p', 'to', 'a ', 'shown', 'make', 'in', 'many', 'immediately', 'wherever', 'the ', 'sub',
    'yourself', 'though', 'everywhere', 'tries', 'yet', 'corresponding', 'besides', 'showns', 'about', 'either',
    'himself', 'much', 'slightly', 'biol', 'such', 'will', 'primarily', 'nt', "that'll", 'through', 'nay', 'relatively',
    'end', 'lest', 'nd', 'indicates', 'become', 'then', 'ca', 'itd', 'possibly', 'que', 'hence', 'downwards',
    'together', 'via', 'inasmuch', 'im', 'believe', 'further', 'usefully', 'an', 'specifying', "they're", 'little',
    'etc', 'edu', 'did', 'are', 'us', 'how', 'their', 'following', 'during', 'own', 'using', 'effect', 'cant', 'the',
    "they'll", "we're", 'know', 'related', 'added', 'yourselves', 'whereas', 'somebody', 'apparently', 'hello',
    'something', 'seven', "n't", 'unto', 'due', 'accordance', 'thus', 'lets', 'anybody', "there've", 'rd', 'although',
    'present', 'containing', 'placed', 'so', 'still', 'thereupon', 'ref', 'really', 'thence', 'meanwhile', 'taken',
    'whomever', 'co', 'does', 'herself', 'second', 'upon', 'auth', 'show', "it's", 'comes', 'considering', 'thorough',
    "isn't", 'ups', 'beforehand', 'behind', 'hither', "hasn't", 'towards', 'affected', 'significantly', 'him', 'qv',
    'hundred', "she'd", 'my', 'wo', 'these', 'announce', 'while', 'says', 'arent', 'nearly', 'cause', 'never', 'ours'
    'by', "she's", 'arise', 'couldnt', 'because', 'contain', 'need', "aren't", 'off', 'certainly', 'has', "who'll",
    'shows', 'should', 'specified', 'thanks', 'new', 'me', 'thereby', 'approximately', 'about ', 'seeing', 'and', 're',
    'hereafter', 'up', 'gives', 'wants', 'whereafter', 'information', 'shed', 'near', 'part', "we'll", 'several',
    'line', "where's", 'wasnt', "you're", "don't", 'thoroughly', 'different', 'way', 'formerly', 'seen', 'plus', 'able',
    'she', 'I ', 'noone', 'as ', 'needs', 'a', 'asking', 'sometime', 'sent', 'stop', 'com ', 'too', 'sorry', 'seeming',
    'obtained', 'ones', 'going', "'d", 'tip', "i'll", 'according', 'latterly', 'your', 'could', 'greetings', 'for ',
    'to ', 'theyd', 'particularly', 'out', "we've", 'showed', 'whim', 'regardless', 'value', 'inc', 'furthermore', 'ie',
    'hereupon', 'thanx', 'whither', 'without', 'predominantly', 'significant', 'for', 'youd', 'vol', 'said', 'e',
    'come', 'specify', 'somethan', 'thru', 'across', 'd', 'hes', 'mrs', 'be', 'any', 'whole', 'theres', 'between',
    'five', "won't", 'lately', "when's", 'was', 'al', 'more', 'ah', 'wouldnt', 'otherwise', 'no', "shan't", 'apart',
    'beyond', 'enough', 'former', 'affects', 'ninety', "that's", 'might', 'proud', 'least', 'available', 'actually',
    'used', 'home', 'would', "he'll", 'certain', 'id', "you'd", 'thou', 'makes', 'similarly', 'themselves', 'put', 'z',
    'yours', 'kg', 'however', 'sec', 'of ', "i'm", 'everybody', 'outside', 'anywhere', 'both', 'every', 'possible',
    'whereupon', 'its', 'others', 'trying', 'indicate', 'somewhat', 'similar', 'brief', 'awfully', 'who', 'looking',
    'follows', 'use', 'thereof', 'kept', 'happens', 'km', 'except', 'research', 'necessarily', 'v', 'we', 'resulted',
    'always', 'seemed', "can't", 'concerning', 'is ', 'thered', 'com', 'wish', 'made', 'meantime', 'whether', 'throug',
    'none', 'resulting', 'youre', 'shall', 'potentially', "what'll", 'probably', 'with', 'maybe', 'zero', 'regarding',
    'even', 'vols', 'quite', 'ml', 'words', 'back', 'q', 'quickly', 'k', 'among', 'another', 'which', 'tends',
    'amongst', 'usually', 'known', 'into', 'overall', "'ll", 'eighty', 'what ', 'aren', "i've", 'it ', 'often', 'sure',
    'th', 'anymore', 'thank', 'begins', 'omitted', 'ord', 'unlike', "'", 'ex', 'afterwards', 'b', 'taking', 'against',
    'eg', 'contains', 'strongly', 'world', 'soon', 'old', 'wont', 'gets', 'whereby', 'insofar', 'done', 'usefulness',
    'one', "let's", "they'd", 'by ', 'entirely', 'theirs', 'this', 'course', 'unfortunately', 'gave', 'less', 'must',
    'abst', 'invention', 'neither', "we'd", 'sometimes', 'being', 'i', "they've", 'novel', 'his', 'hers', 'think',
    'fifth', 'tell', 'especially', 'nor', 'x', 'appear', 'readily', 'four', 'mug', 'largely', 'becoming', 'getting',
    'instead', 'obtain', 'willing', 'sufficiently', 'widely', 'few', 'yes', 'www', 'name', 'previously', "you'll",
    'owing', "'re", 'our', 'see', 'page', 'secondly', 'hopefully', "couldn't", 'some', "mustn't", 'u', 'down', 'from',
    'best', 'hardly', 'useful', 'poorly', 'like', 'indicated', 'thoughh', 'having', 'unless', 'whoever', 'til', '-',
    'go', 'took', 'in ', 'those', 'here', 'saw', 'are ', 'I'
}

