import pandas as pd
from langdetect import detect
from googletrans import Translator, constants
from empath import Empath
import string
import numpy as np
PUNCUATION_LIST = list(string.punctuation)

RESEARCH_QUESTION = '5575c3ac-ad3c-4cab-adc1-24bbf02f45d8'

def load_dataframe(filepath: str) -> pd.DataFrame:
    print(f'Loading dataframe from {filepath}')
    return pd.read_csv(filepath)

lexicon = Empath()
SIZE=300
c=lexicon.create_category("computer", ["programming", "informatics", "computer"], model="nytimes", size=SIZE)
c=lexicon.create_category("history", ["history", "History"], model="nytimes", size=SIZE)
c=lexicon.create_category("chemistry", ["chemistry", "biochemistry", "chemical_reaction", "atom", "organic_chemistry", "hydrogen", "distill"],  model="nytimes", size=SIZE)
c=lexicon.create_category("geography", ["geography", "map", "country"], model="nytimes", size=SIZE)
c=lexicon.create_category("math", ["math", "number", "equation", "arithmetic", "calculus", "algebra"], model="nytimes", size=SIZE)
c=lexicon.create_category("literature", ["literature", "book", "author", "write", "spelling", "literary", "reading", "read"], model="nytimes", size=SIZE)
c=lexicon.create_category("physics", ["physics", "force", "magnet", "quantum", "mechanics"], model="nytimes", size=SIZE)
c=lexicon.create_category("biology", ["biology", "organism"], model="nytimes", size=SIZE)
translator = Translator()

def get_translation(title):
  try:
      prep = translator.translate(title).text
  except Exception:
      prep = ""
  return prep

def get_topic(title):
    try:
      prep = translator.translate(title).text
    except Exception:
      prep = ""

    prep = prep.lower()
    prep = " ".join([w for w in prep.split() if w not in PUNCUATION_LIST])
    for char in PUNCUATION_LIST:
        prep = prep.replace(char, " ")
    if len(prep)==0:
        return np.nan
    res = lexicon.analyze(prep, categories = ["computer", "history", "chemistry", "geography", "math", "literature", "physics", "biology"], normalize = True)
    try:
      m = max(res, key=res.get)
    except Exception:
      m = 0
    return m if res[m]>0 else np.nan
    

def process_time_series_data(data_dir: str, output: str) -> None:
    # Load reflection answers raw data
    reflection_answers = pd.read_csv('{}/classtime_reflection_answers.csv.gz'.format(data_dir), 
                                     usecols=['participant_id', 'reflection_question_id', 'value'])
    # Only keep relevant reflection answers and keep last answer in case of duplicates
    reflection_answers = reflection_answers[reflection_answers.reflection_question_id == RESEARCH_QUESTION]\
                      .drop_duplicates(subset=['participant_id'], keep='last')
    
    # Participants to consider
    answering_participants = reflection_answers.participant_id
    
    # Load answers data
    answers = []
    with pd.read_csv('{}/classtime_answers.csv.gz'.format(data_dir), chunksize=1000000, 
                 usecols=['participant_id', 'question_id', 'session_id', 'is_correct', 
                          'rating', 'created_at']) as df_reader:
        for chunk in df_reader:
            # Only keep relevant participant answers
            answers.append(chunk[chunk.participant_id.isin(answering_participants)])
    # Concatenate filtered chunks and keep last answer in case of duplicate participant answers on a question
    answers = pd.concat(answers).drop_duplicates(subset=['participant_id', 'question_id'], keep='last')\
                .sort_values(by=['participant_id', 'created_at'])
    answers = answers.iloc[:1_000_0]
    answers['created_at'] = answers.created_at.mul(-1) # Prepare column for future calculation
    print(len(answers))
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
    
    # Load sessions data
    sessions = pd.read_csv('{}/classtime_sessions.csv.gz'.format(data_dir), 
                           usecols=['id', 'mode', 'feedback_mode', 'force_reflection', 'timer', 'is_solo', 'title', 'is_onboarding'])
    # Keep only relevant sessions
    sessions = sessions[sessions.id.isin(answers.session_id.unique())]
    # TODO drop onloading sessions
    # Get session information (drop duplicates and keep latest update)
    session_information = sessions.drop_duplicates(subset=['id'], keep='last').set_index('id')
    session_information["title_len"] = session_information["title"].str.len()
    session_information['title_topic'] = session_information["title"].apply(lambda t: get_topic(t))
    session_information['title_trans'] = session_information["title"].apply(lambda t: get_translation(t))

    # Add respective columns to the answers dataframe
    answers['mode'] = answers.session_id.map(session_information['mode'])
    answers['feedback_mode'] = answers.session_id.map(session_information.feedback_mode)
    answers['force_reflection'] = answers.session_id.map(session_information.force_reflection)
    answers['timer'] = answers.session_id.map(session_information.timer)
    answers['is_solo'] = answers.session_id.map(session_information.is_solo)
    # Features extracted from title
    answers['title_lang'] = answers.session_id.map(session_information.title).apply(lambda t: detect(t) if (not isinstance(t, float) and len(''.join([c for c in t.split() if c.isalpha()]))>0) else np.nan)
    answers['title_len'] = answers.session_id.map(session_information.title_len)
    answers['title_topic'] = answers.session_id.map(session_information.title_topic)
    answers['title_trans'] = answers.session_id.map(session_information.title_trans)

    # Session ID no longer needed in answers
    answers.drop(labels=['session_id'], axis='columns', inplace=True)
    
    # Load questions data
    questions = pd.read_csv('{}/classtime_questions.csv.gz'.format(data_dir), 
                        usecols=['id', 'weight', 'video', 'image'])
    # Keep only relevant questions
    questions = questions[questions.id.isin(answers.question_id.unique())]
    # Get questions information (drop duplicates and keep latest update)
    question_information = questions.drop_duplicates(subset=['id'], keep='last').set_index('id')
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
        .combine_first(answers.is_correct.map({'t': 1.0, 'f': 0.0}, na_action='ignore')).fillna(-1.0)
    # Drop no longer needed column
    answers.drop(labels=['is_correct'], axis='columns', inplace=True)
    
    # Add participant n-th answer number column
    answers['nth_answer'] = answers.groupby('participant_id').cumcount().reset_index(drop=True)
    
    # Add answer to the feeling of learning research question
    fol_answer = reflection_answers.groupby('participant_id').value.last()
    answers['response'] = answers.participant_id.map(fol_answer)
    
    # Replace invalid data to np.nan
    answers.replace(to_replace=-1.0, value=np.nan, inplace=True)
    
    # Save processed dataframe
    print(f'Saving processed data at {output}')
    answers.to_csv(output, index=False, compression='gzip')