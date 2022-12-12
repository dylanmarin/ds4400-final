from pickle import load
import pandas as pd
from keras_preprocessing.text import Tokenizer

import gc
from keras.backend  import set_session, clear_session, get_session
import tensorflow as tf
from nltk.translate.bleu_score import corpus_bleu

RANDOM_SEED = 42

'''
get the filenames for the training, validation, and test sets
'''
# get the specific set filenames from the dataset 
ALL_FILENAMES = list(set(pd.read_csv('../data/flickr_8k/captions.txt')['image']))
TRAIN_FILENAMES = list(set(pd.read_csv('../data/flickr_8k/train.csv')['image']))
TEST_FILENAMES = list(set(pd.read_csv('../data/flickr_8k/test.csv')['image']))
VALIDATION_FILENAMES = list(set(pd.read_csv('../data/flickr_8k/validation.csv')['image']))
TRAIN_AND_VAL_FILENAMES = list(set(pd.read_csv('../data/flickr_8k/train_and_val.csv')['image']))
SMALL_TRAIN_FILENAMES = list(set(pd.read_csv('../data/flickr_8k/small_train.csv')['image']))


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
    output = [START_TOK] + output + [END_TOK]

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


def reset_keras(model):
    '''
    utility function to reset gpu memory
    
    Code from:
    https://github.com/keras-team/keras/issues/12625
    https://forums.fast.ai/t/how-could-i-release-gpu-memory-of-keras/2023/18
    '''
    sess = get_session()
    clear_session()
    sess.close()
    sess = get_session()

    try:
        del model
    except:
        pass

    gc.collect()
    print('Garbage collected.')

    config = tf.compat.v1.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 1
    config.gpu_options.visible_device_list = "0"
    set_session(tf.compat.v1.Session(config=config))
    
    
def corpus_bleu_score(all_original_captions, all_generated_captions):
    '''
    Given a list of corresponding captions to images (# samples x 5 captions per image (where each caption is a list of strings)),
    and the model's generated captions for each image:
        output a tuple of corpus BLEU scores (1 through 4)
        
    Removes <start> and <end> from all original captions (all generated captions shouldnt have start and end)
    '''
    # prepare to remove start and end from all original captions
    cleaned_original_captions = []
    
    # for each caption group (of 5 corresponding captions)
    for caption_group in all_original_captions:
        captions_to_add = []
        
        # for each caption
        for caption in caption_group:
            # remove the first and last token and append it
            caption = caption[1:len(caption)-1]
            captions_to_add.append(caption)
            
        cleaned_original_captions.append(captions_to_add)
    
    # printing final model score
    bleu_1 = corpus_bleu(cleaned_original_captions, all_generated_captions, weights=(1.0, 0, 0, 0))
    bleu_2 = corpus_bleu(cleaned_original_captions, all_generated_captions, weights=(0.5, 0.5, 0, 0))
    bleu_3 = corpus_bleu(cleaned_original_captions, all_generated_captions, weights=(1/3, 1/3, 1/3, 0))
    bleu_4 = corpus_bleu(cleaned_original_captions, all_generated_captions, weights=(0.25, 0.25, 0.25, 0.25))
    
    print(f'BLEU-1: {bleu_1}')
    print(f'BLEU-2: {bleu_2}')
    print(f'BLEU-3: {bleu_3}')
    print(f'BLEU-4: {bleu_4}')
    
    return bleu_1, bleu_2, bleu_3, bleu_4    