from pickle import load
import pandas as pd
from sklearn.model_selection import train_test_split
from keras_preprocessing.text import Tokenizer


RANDOM_SEED = 42

'''
get the filenames for the training, validation, and test sets

split the train/val/test 80/10/10
'''
# get all of the filenames from the dataset 
ALL_FILENAMES = list(set(pd.read_csv('data/flickr_8k/captions.txt')['image']))
# split the filenames into train test val 80-10-10
TRAIN_FILENAMES, TEST_FILENAMES = train_test_split(ALL_FILENAMES, test_size=0.2, random_state=RANDOM_SEED)
TEST_FILENAMES, VALIDATION_FILENAMES = train_test_split(TEST_FILENAMES, test_size=0.5, random_state=RANDOM_SEED) 
# define a combined training plus validatoin set
TRAIN_AND_VAL_FILENAMES = TRAIN_FILENAMES + VALIDATION_FILENAMES

START_TOK = '<start>'
END_TOK = '<end>'

def clean_string(text):
    '''
    returns new array of tokens representing the text

    - lowercased
    - removes 1 - letter punctuation
    - removes numbers
    - appends 's to previous words
    - reconstructs string

    <start> is appended to the start
    <end> is appended to the end

    Notes:
    maybe keep in numbers
    maybe remove all dashes 
    '''
    output = []

    text = text.lower().replace('"', '')
    
    tokens = text.split()
    for token in tokens:
        # keep words that are only alphabet characters
        # exclude any non-alphabetic only words that are only 1 character long (punctuation)
        # remove any numbers 
        if token.isalpha() or ((not token.isalpha() and len(token) > 1) and not token.isnumeric()):
            output.append(token)

    # dataset contained the string "'s" separated by whitespace
    # this reappends that string to the word before it
    for i, token in enumerate(output):
        if token == "'s":
            output[i-1] = output[i-1] + "'s"
            output.remove("'s")

        if len(token) == 2 and '.' in token:
            output[i] = token.replace('.', '')
    
    # add a start and end token that will be used for generation
    output = ['<start>'] + output + ['<end>']

    return output

def clean_descriptions(filename):
    '''
    from the given file import the data as a csv and clean the captions
    '''
    data = pd.read_csv(filename)
    data['caption'] = data['caption'].apply(lambda caption: clean_string(caption))
    return data


def samples_to_dict(samples):
    '''
    given a dataframe with 'image' and 'caption' columns

    return a dictionary where:
        key = filename (from image column)
        value = list of all samples (each sample is a list of tokens)
    '''
    descriptions = dict()
    for image, caption in zip(samples['image'], samples['caption']):
        if image not in descriptions.keys():
            # initalize the dictionary entry
            descriptions[image] = [caption]
        else:
            # append to the dictionary entry
            descriptions[image].append(caption)	

    return descriptions


def import_image_features(features_file, corresponding_filenames):
    '''
    from our stored pkl file of extracted VGG features,
    given the pkl file name, and a list of photo filenames
    return a dictionary from filename to VGG features
    '''
    # import all features from pkl file
    all_features = load(open(features_file, 'rb'))
    
    # get a dictionary from filename to image features
    # splits filename at '.' because the pkl file doesnt store the .jpg part of the filename
    features = {filename: all_features[filename.split('.')[0]] for filename in corresponding_filenames}
    
    return features

def max_and_average_sequence_length(sequences):
    '''
    given a df of filenames and captions (tokenized into lists of strings)
    return the max and average length of the captions

    also prints information about them as well
    '''
    sequence_lengths = list(sequences['caption'].apply(lambda caption : len(caption)))
    sequence_lengths.sort(reverse=True)
    MAX_LENGTH = max(sequence_lengths)
    AVG_LENGTH = int(sum(sequence_lengths) / len(sequence_lengths))
    print(f'The top 30 sequence lengths are:\n{sequence_lengths[:30]}')
    print(f'The longest sequence length from the training and validation samples is {MAX_LENGTH}')
    print(f'The average sequence length from the training and validation samples is {AVG_LENGTH}')
    return MAX_LENGTH, AVG_LENGTH


def get_tokenizer_from_samples(samples):
    '''
    given a dataframe of samples (where the caption column is tokenized captions), 
    create a tokenizer from the given captions 
    '''
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(list(samples['caption']))
    return tokenizer