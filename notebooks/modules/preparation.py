import os
import ast
import numpy as np
import pandas as pd

from empath import Empath
# from google.cloud import translate_v2 as translate

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler, PolynomialFeatures
from imblearn.under_sampling import RandomUnderSampler

import string
PUNCTUATION_LIST = list(string.punctuation)

RESEARCH_QUESTION = '5575c3ac-ad3c-4cab-adc1-24bbf02f45d8'

DATA_DIR = '../data'
RAW_DATA = f'{DATA_DIR}/raw'
INTERIM_DATA = f'{DATA_DIR}/interim'
PROCESSED_DATA = f'{DATA_DIR}/processed'

def keep_relevant():
    '''Only keep participants and sessions relevant to the answered FOL question'''
    # First load raw feeling of learning answers data
    reflection_answers = pd.read_csv(f'{RAW_DATA}/classtime_reflection_answers.csv.gz',
                            usecols=['participant_id', 'session_id', 'reflection_question_id', 'value'])
    
    # Only keep relevant reflection answers and keep last answer in case of duplicates
    reflection_answers = reflection_answers[reflection_answers.reflection_question_id == RESEARCH_QUESTION]\
                          .drop_duplicates(subset=['participant_id'], keep='last')
    
    # Load sessions data
    sessions = pd.read_csv(f'{RAW_DATA}/classtime_sessions.csv.gz', 
                           usecols=['id', 'title', 'mode', 'feedback_mode', 'force_reflection', 'timer', 'is_solo', 'is_onboarding'])
    
    # Drop onboarding sessions
    sessions = sessions[sessions.is_onboarding == False]
    # Drop onboarding column
    sessions.drop(labels=['is_onboarding'], axis='columns', inplace=True)
    # Rename ID column to participant_id
    sessions.rename(columns={'id': 'session_id'}, inplace=True)
    # Keep only sessions with participants that have answered relevant question
    sessions = sessions[sessions.session_id.isin(reflection_answers.session_id)]
    
    # Drop duplicates and keep last to consider latest update
    sessions.drop_duplicates(subset='session_id', keep='last', inplace=True)    
    # Save relevant sessions and attached information
    sessions.to_csv(f'{INTERIM_DATA}/relevant_sessions.csv.gz', index=False, compression='gzip')
    
    # Drop participants that were in onboarding sessions
    reflection_answers = reflection_answers[reflection_answers.session_id.isin(sessions.session_id)]
    # Save considered participants and their answer
    answering_participants = reflection_answers[['participant_id', 'session_id', 'value']].rename(columns={'value': 'response'})
    answering_participants.to_csv(f'{INTERIM_DATA}/answering_participants.csv.gz', index=False, compression='gzip')
    
    
def filter_answers_questions():
    '''Keep relevant participant answers'''
    # Load answering participants
    answering_participants = pd.read_csv(f'{INTERIM_DATA}/answering_participants.csv.gz', usecols=['participant_id'])
    
    # Load answers data
    answers = []
    with pd.read_csv(f'{RAW_DATA}/classtime_answers.csv.gz', chunksize=1000000, 
                 usecols=['participant_id', 'question_id', 'session_id', 'is_correct', 
                          'rating', 'created_at']) as df_reader:
        for chunk in df_reader:
            # Only keep relevant participant answers
            answers.append(chunk[chunk.participant_id.isin(answering_participants.participant_id)])
            
    # Concatenate filtered chunks and keep last answer in case of duplicate participant answers on a question
    answers = pd.concat(answers)
    answers.drop_duplicates(subset=['participant_id', 'question_id'], keep='last', inplace=True)
    answers.sort_values(by=['participant_id', 'created_at'], inplace=True)
    
    # Save filtered answers
    answers.to_csv(f'{INTERIM_DATA}/filtered_answers.csv.gz', index=False, compression='gzip')
    
    # Load questions data
    questions = pd.read_csv(f'{RAW_DATA}/classtime_questions.csv.gz', 
                           usecols=['id', 'weight', 'video', 'image'])
    # Keep only relevant questions
    questions = questions[questions.id.isin(answers.question_id.unique())]
    # Drop duplicates and keep last to consider latest update
    questions.drop_duplicates(subset='id', keep='last', inplace=True)
    # Rename id column to question_id
    questions.rename(columns={'id': 'question_id'}, inplace=True)
    
    # Save filtered questions
    questions.to_csv(f'{INTERIM_DATA}/filtered_questions.csv.gz', index=False, compression='gzip')

    
def translate_titles():
    '''Translate titles using Google Cloud Translate API'''
    # Read sessions
    sessions = pd.read_csv(f'{INTERIM_DATA}/relevant_sessions.csv.gz', 
                           usecols=['session_id', 'title']).set_index('session_id')
    sessions.rename(columns={'title': 'translation'}, inplace=True)
    
    # Instantiate client and provide credentials
    os.environ['GOOGLE_APPLICATION_CREDENTIALS']='my-translation-sa-key.json'
    translate_client = translate.Client()
    
    N_SESSIONS = len(sessions.index)
    
    # Translate in batches of 10'000
    for i in range(int(np.ceil(N_SESSIONS / 10000.))):
        # Translate batch
        ts = sessions.iloc[i*10000:min(N_SESSIONS, (i+1)*10000)].translation.map(
            lambda t: translate_client.translate(t))
        # Save batch
        ts.to_csv(f'{INTERIM_DATA}/translations/translated-{i}', compression='gzip')
        

def get_topic(text, lexicon):
    '''Helper: Get text topic'''
    # Apply lowercase to text
    foo = text.lower()
    # Extract words
    foo = ' '.join([w for w in foo.split() if w not in PUNCTUATION_LIST])
    # Replace punctuation by whitespace
    for char in PUNCTUATION_LIST:
        foo = foo.replace(char, ' ')
    
    # Analyze categorical resemblance
    res = lexicon.analyze(foo, categories = ["computer", "history", "chemistry", "geography", "math",
                                              "literature", "physics", "biology"], normalize = True)
    try:
        m = max(res, key=res.get) # Find category with highest resemblance
        res = m if res[m]>0 else np.nan # Only select category if resemblance > 0
    except Exception:
        res = np.nan
    return res
    
        
def augment_session_info():
    '''Augment session information with extra features'''
    # Load relevant sessions data
    sessions = pd.read_csv(f'{INTERIM_DATA}/relevant_sessions.csv.gz')
    
    # Load translation data (which was split in batches of 10k)
    n_batches = int(np.ceil(len(sessions)/10000.))
    translations = [pd.read_csv(f'{INTERIM_DATA}/translations/translated-{i}', 
                                compression='gzip') for i in range(n_batches)]
    translations = pd.concat(translations).set_index('session_id')
    # Evaluate column as dictionary
    translations['translation'] = translations.translation.apply(ast.literal_eval)
    
    # Add title length
    sessions['title_len'] = sessions.title.map(lambda t: len(t))
    # Add title language
    sessions['title_lang'] = translations.translation.map(lambda t: t['detectedSourceLanguage']).values
    # Add title translation
    sessions['title_trans'] = translations.translation.map(lambda t: t['translatedText']).values
    
    # Generate categories to extract likely topic
    lexicon = Empath()
    SIZE=300
    c=lexicon.create_category("computer", ["programming", "informatics", "computer"], 
                              model="nytimes", size=SIZE)
    c=lexicon.create_category("history", ["history", "History"], 
                              model="nytimes", size=SIZE)
    c=lexicon.create_category("chemistry", ["chemistry", "biochemistry", "chemical_reaction", "atom", "organic_chemistry", "hydrogen", "distill"],  
                              model="nytimes", size=SIZE)
    c=lexicon.create_category("geography", ["geography", "map", "country"], 
                              model="nytimes", size=SIZE)
    c=lexicon.create_category("math", ["math", "number", "equation", "arithmetic", "calculus", "algebra"], 
                              model="nytimes", size=SIZE)
    c=lexicon.create_category("literature", ["literature", "book", "author", "write", "spelling", "literary", "reading", "read"], 
                              model="nytimes", size=SIZE)
    c=lexicon.create_category("physics", ["physics", "force", "magnet", "quantum", "mechanics"], 
                              model="nytimes", size=SIZE)
    c=lexicon.create_category("biology", ["biology", "organism"], 
                              model="nytimes", size=SIZE)
    
    # Add title category
    sessions['title_topic'] = sessions.title_trans.map(lambda t: get_topic(t, lexicon))
    
    # Save augmented session information
    sessions.set_index('session_id').to_csv(f'{INTERIM_DATA}/session_information.csv.gz', compression='gzip')
    
def augment_answers():
    '''Augment relevant answers data with added features'''
    # Load relevant answers data
    answers = pd.read_csv(f'{INTERIM_DATA}/filtered_answers.csv.gz')
    # Load session information data
    session_information = pd.read_csv(f'{INTERIM_DATA}/session_information.csv.gz', index_col='session_id')
    # Load question information data
    question_information = pd.read_csv(f'{INTERIM_DATA}/filtered_questions.csv.gz', index_col='question_id')
    # Load participant feeling of learning responses data
    reflection_answers = pd.read_csv(f'{INTERIM_DATA}/answering_participants.csv.gz', index_col='participant_id')

    # Prepare column for future calculation
    answers['created_at'] = answers.created_at.mul(-1)
    # Group answers by participant
    participant_answers = answers.groupby('participant_id', sort=False)
    # Calculate participant answer times
    timings = participant_answers.created_at.transform(lambda g: g.diff(-1)).reset_index(drop=True)
    # Calculate max number of questions answered by a participant
    MAX_ANSWERED = participant_answers.size().max()
    # Drop created_at column that is no longer useful
    answers = participant_answers.head(MAX_ANSWERED).reset_index(drop=True)\
                .drop(labels=['created_at'], axis='columns')
    # Add answer_time column to answers
    answers['answer_time'] = timings.fillna(-1.0)

    # Add respective session information columns to the answers dataframe
    answers['mode'] = answers.session_id.map(session_information['mode'])
    answers['feedback_mode'] = answers.session_id.map(session_information.feedback_mode)
    answers['force_reflection'] = answers.session_id.map(session_information.force_reflection)
    answers['timer'] = answers.session_id.map(session_information.timer)
    answers['is_solo'] = answers.session_id.map(session_information.is_solo)
    # Features extracted from title
    answers['title_lang'] = answers.session_id.map(session_information.title_lang).fillna('unknown')
    answers['title_len'] = answers.session_id.map(session_information.title_len).fillna(0)
    answers['title_topic'] = answers.session_id.map(session_information.title_topic).fillna('unknown')
    # Session ID no longer needed in answers
    answers.drop(labels=['session_id'], axis='columns', inplace=True)

    # Add respective columns to the answers dataframe
    answers['weight'] = answers.question_id.map(question_information.weight)
    answers['video'] = answers.question_id.map(question_information.video)
    answers['image'] = answers.question_id.map(question_information.image)
    # Question ID no longer needed in answers
    answers.drop(labels=['question_id'], axis='columns', inplace=True)

    # Update rating and weight columns to represent correctness percentage
    answers['correctness'] = answers.rating / answers.weight
    # Drop columns no longer needed
    answers.drop(labels=['rating', 'weight'], axis='columns', inplace=True)

    # Combine is_correct and correctness columns
    answers['correctness'] = answers.correctness \
        .map(lambda x: np.nan if x > 1.0 or x < 0.0 else x, na_action='ignore') \
        .combine_first(answers.is_correct.map({'t': 1.0, 'f': 0.0}, na_action='ignore'))
    # Drop no longer needed column
    answers.drop(labels=['is_correct'], axis='columns', inplace=True)

    # Add participant n-th answer number column
    answers['nth_answer'] = answers.groupby('participant_id').cumcount().reset_index(drop=True)

    # Add answer to the feeling of learning research question
    answers['response'] = answers.participant_id.map(reflection_answers.response)

    # Save processed dataframe
    answers.to_csv(f'{PROCESSED_DATA}/augmented_answers.csv.gz', index=False, compression='gzip')
    
def nan_columns():
    '''Returns list of column names which contain nan values'''
    # Load data
    answers = pd.read_csv(f'{PROCESSED_DATA}/augmented_answers.csv.gz')
    # Return column names that contain invalid/missing data
    return answers.columns[answers.isna().any()].to_list()

def impute(cat_cols, num_cols):
    '''Impute missing data'''
    # Load data
    answers = pd.read_csv(f'{PROCESSED_DATA}/augmented_answers.csv.gz')
    
    # Instantiate imputers
    imp_frequent = SimpleImputer(strategy='most_frequent')
    imp_mean = SimpleImputer(strategy='mean')
    
    # Impute categorical columns
    for col in cat_cols:
        answers[col] = imp_frequent.fit_transform(answers[col].array.reshape(-1, 1))
    
    # Impute numerical columns
    for col in num_cols:
        answers[col] = imp_mean.fit_transform(answers[col].array.reshape(-1, 1))
    
    # Save data with imputed values
    answers.to_csv(f'{INTERIM_DATA}/imputed_answers.csv.gz', index=False, compression='gzip')

def encode_categorical():
    '''Encode categorical data'''
    # Load data
    answers = pd.read_csv(f'{INTERIM_DATA}/imputed_answers.csv.gz')
    # Categorical columns to encode
    cat_cols = ['mode', 'feedback_mode', 'force_reflection', 'is_solo', 'title_lang', 'title_topic', 'video', 'image']
    
    # Encode
    for col in cat_cols:
        answers[col] = answers[col].astype('category').cat.codes.astype('float')
    
    # Save intermediate result
    answers.to_csv(f'{INTERIM_DATA}/categories_encoded.csv.gz', index=False, compression='gzip')

def normalize():
    '''Normalize data'''
    # Load data
    answers = pd.read_csv(f'{INTERIM_DATA}/categories_encoded.csv.gz')
    # Instantiate scaler
    min_max = MinMaxScaler()
    # Columns to scale (all but response, participant_id, and nth_answer)
    columns = ['answer_time', 'mode', 'feedback_mode',
       'force_reflection', 'timer', 'is_solo', 'title_lang', 'title_len',
       'title_topic', 'video', 'image', 'correctness']
    
    # Normalize
    for col in columns:
        answers[col] = min_max.fit_transform(answers[col].values.reshape(-1, 1))
        
    # Save dataset
    answers.to_csv(f'{PROCESSED_DATA}/final_data.csv.gz', index=False, compression='gzip')
    
def aggregate_participant_data():
    '''Generate by-participant aggregate dataset'''
    # Load dataset to aggregate
    answers = pd.read_csv(f'{PROCESSED_DATA}/final_data.csv.gz')
    
    # Aggregate data by participant
    aggregated = answers.groupby('participant_id').agg({
        'participant_id': 'first',
        'answer_time': ['min', 'max', 'mean', 'std'],
        'mode': 'first',
        'feedback_mode': 'first',
        'force_reflection': 'first',
        'timer': 'first',
        'is_solo': 'first',
        'title_lang': 'first',
        'title_len': 'first',
        'title_topic': 'first',
        'video': 'mean',
        'image': 'mean',
        'correctness': ['mean', 'std'],
        'nth_answer': 'size', # Number of questions answered
        'response': 'first'
    })
    
    # Flatten index
    aggregated.columns = ["_".join(a[::-1]).replace('first_', '') for a in aggregated.columns.to_flat_index()]
    # Rename column
    aggregated.rename(columns={'size_nth_answer': 'n_answers'}, inplace=True)
    
    # Set undefined std values to 0
    aggregated['std_answer_time'] = aggregated.std_answer_time.fillna(0)
    aggregated['std_correctness'] = aggregated.std_correctness.fillna(0)
    
    # Save dataset
    aggregated.to_csv(f'{PROCESSED_DATA}/participants_aggregated.csv.gz', index=False, compression='gzip')
    
def make_balanced_datasets():
    '''Generate balanced datasets for the multiclass and binary prediction task'''
    # Load data
    aggregated = pd.read_csv(f'{PROCESSED_DATA}/participants_aggregated.csv.gz')
    
    # Get labels
    y_multiclass = aggregated.pop('response')
    y_binary = y_multiclass.map({'happy': 'happy', 'neutral': 'not_happy', 'upset': 'not_happy'})
    
    # Features
    X = aggregated
    
    # Instantiate balancer
    rus = RandomUnderSampler(random_state=123)
    
    # Balance datasets
    X_multi, y_multi = rus.fit_resample(X, y_multiclass)
    X_bin, y_bin = rus.fit_resample(X, y_binary)
    
    # Concatenate labels with features
    multi = pd.concat([X_multi, y_multi], axis=1)
    binary = pd.concat([X_bin, y_bin], axis=1)
    
    # Save balanced datasets
    multi.to_csv(f'{PROCESSED_DATA}/balanced_multi_agg.csv.gz', index=False, compression='gzip')
    binary.to_csv(f'{PROCESSED_DATA}/balanced_bin_agg.csv.gz', index=False, compression='gzip')
    
def prepare_time_series_data(n_steps):
    '''Prepare dataset for time series prediction with tensorflow'''
    # Load the dataset
    df = pd.read_csv(f'{PROCESSED_DATA}/final_data.csv.gz')
    
    # Keep first n_steps answers by participant
    df = df[df.nth_answer < n_steps]
    # Extract labels
    labels = df.groupby('participant_id').response.first()
    # Drop unused columns
    df.drop(labels=['nth_answer', 'response'], axis='columns', inplace=True)
    
    # Number of features = # of columns - participant_id column
    N_FEATURES = df.shape[1] - 1
    feature_cols = df.columns.values[1:]
    
    # Since not all participants have a enough answers, we need to pad our data
    PAD_VALUE = -1.0

    # Helper padding function
    def pad(values, n_steps=n_steps, pad_val=PAD_VALUE):
        return np.pad(values, [(0, n_steps-values.shape[0]), (0, 0)], mode='constant', constant_values=pad_val)

    # Add padding
    df = df.groupby('participant_id').apply(lambda r: np.stack(pad(r[feature_cols].values), axis=0)).explode()

    # Explode column of list
    X = np.array(df.to_list())
    # Reshape as tensor
    X = X.reshape(-1, n_steps, N_FEATURES)

    # Save
    np.save(f'{PROCESSED_DATA}/{n_steps}-steps.npy', X)
    labels.to_csv(f'{PROCESSED_DATA}/participant-labels.csv.gz', compression='gzip')