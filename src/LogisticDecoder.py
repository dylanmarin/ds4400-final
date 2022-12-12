from common import START_TOK, END_TOK, reset_keras, RANDOM_SEED, import_image_features, ALL_FILENAMES
import numpy as np
from tensorflow.keras.layers import ReLU
from tensorflow.keras.layers import Dense
from keras.models import Sequential
from keras.models import load_model
import tensorflow as tf
from tensorflow.keras.utils import to_categorical

all_image_features = import_image_features('../data8k_features.pkl', ALL_FILENAMES)

def get_samples_of_specific_size(image_vector, descriptions, desired_caption_size, tokenizer):
    '''
    Given:
    - one image vector representing a single image
    - one value from our samples dict for the corresponding image (a list of 5 tokenized captions)
    - the desired caption size N
    - a Tokenizer

    Return X:
        a tensor with 5 elements (one for each caption) where each item is an array with length 4096 + N
        - the first 4096 elements are the VGG extracted features of the corresponding image
        - the next N elements are the first N words of the caption (converted to numbers by the passed in tokenizer)
    And Y:
        a tensor with 5 elements where each element is:
        - the N+1 word in the sequence

    NOTE: If any caption has a total length less than or equal to N, then it will not be added to the output, 
    meaning that the output could be tensors with 0 dimensions
    '''
    # initalize empty arrays for the samples
    X = []
    y = []

    # reshape the image vector to be one dimension
    image_vector = image_vector.reshape(-1,)
                                
    # convert the descriptions to number lists instead of string lists
    descriptions = tokenizer.texts_to_sequences(descriptions)
    
    # get vocab size from tokenizer
    vocab_size = len(tokenizer.index_word) + 1

    # for each description
    for description in descriptions:
        # only if the caption is at least 1 longer than N (desired_caption_size)
        if len(description) >= desired_caption_size + 1:
            # get the caption as a numpy array of elements (length will be desired caption size)
            caption = np.array(description[:desired_caption_size])
        
            # concatenate the image vector with the caption vector
            combined_X = np.concatenate([image_vector, caption])

            # append the combined X vector to the output x list
            X.append(combined_X)

            # get the one-hot encoding of the last word (required for keras models)
            # and append it to the output y list
            last_word = description[desired_caption_size]
            
            y.append(to_categorical(last_word, vocab_size))

    return tf.convert_to_tensor(X), tf.convert_to_tensor(y)


def data_generator(filename_description_dictionary, desired_caption_size, loops, tokenizer):
    '''
    Given:
    - A dictionary containing the samples we want to create a generator for where
        key: filename (string)
        value: list of captions (where each caption is a list of strings)
    - A desired caption size (N)
    - A number of loops L
    - A tokenizer for converting seen words to numbers
    
    loops are used because a generator is expended once it yields its last result, and 
    therefore cannot be used over multiple epohchs

    Iterate through the filename_description dictionary (L times).
    For each filename, generate corresponding number of samples where the caption size is N
    Each value in the X samples vector will be 4096 + N
        Those N values are the first N words of the corresponding caption
    Each value in the y vector will be the N+1 word

    Used to save memory
    Each loop it shuffles the order of the samples
    '''
    for _ in range(loops):
        # shuffle filename order for better distribution over multiple loops (epochs)
        np.random.seed(RANDOM_SEED)
        all_filenames = list(filename_description_dictionary.keys())
        np.random.shuffle(all_filenames)
        
        # loop for ever over filenames
        for filename in all_filenames:
            # get the corresponding descriptions
            descriptions = filename_description_dictionary[filename]

            # retrieve the image feature vector
            image_vector = all_image_features[filename][0]

            # get the samples of the desired size (N) 
            x_samples, y_samples = get_samples_of_specific_size(image_vector, descriptions, desired_caption_size, tokenizer)
            
            # if there are no samples of that shape
            if y_samples.shape == (0,):
                # continue the loop until there are
                continue
            
            yield x_samples, y_samples
            
            
def generate_logistic_model(input_size, output_size):
    '''
    generate a logistic regression model using keras api
    
    since our model uses multiple logistic regression models, 
    we wanted to run it on the gpu which is simple with keras
    '''
    # create a linear activation function, relu which doesn't punish values < 0
    linear_activation = ReLU(negative_slope=1)

    # FF NN
    model = Sequential()

    # input layer is given input size (4096 + number of words for corresponding decoder)
    model.add(Dense(input_size, activation=linear_activation))

    # output layer with softmax for the whole vocabulary
    model.add(Dense(output_size, activation='softmax'))

    # compile and return
    model.compile(loss='categorical_crossentropy', optimizer='adam')
    return model


class LogisticDecoder():
    '''
    this class represents a model that can generate captions for text based on image input

    it works by creating a logistic regression classifier for each position in the output string

    each logistic regression model is assigned an index corresponding to the number of words it takes as input
    for example:
        model at index 3 is responsible for taking in a feature vector of length 4096 + 3
        the first 4096 values are the image input, and the +3 represents the first 3 strings in the caption
        the model predicts the 4th word
    '''

    def __init__(self, caption_max_length, tokenizer):
        '''
        given a max caption length N, initalize N logistic regression models, one for each position in our caption length
        given a tokenizer, store it in the model
        '''
        self.max_len = caption_max_length

        # store the tokenizer for later use
        self.tokenizer = tokenizer
        
        # get the vocab size (add one due to the way keras tokenizer works)
        self.vocab_size = len(tokenizer.word_index) + 1

        # generate a model that takes in an image feature vector, and the caption so far, and outputs the next word
        self.models = [None for i in range(caption_max_length)]


    def fit(self, sample_dictionary, epochs, model_save_directory, verbose=False):
        '''
        given a dictionary of samples (key is a filename and value is all associated captions tokenized into lists of strings)
        and a number of epochs

        train the logistic decoders to generate captions

        if model_save_directory is given, save the logistic models into the given directory
        don't add a / at the end of the directory
        '''
        for i in range(self.max_len):
            if verbose:
                print(f'Training model #{i+1}')

            current_generator = data_generator(sample_dictionary, desired_caption_size=i+1, loops=epochs, tokenizer=self.tokenizer)
            
            current_model = generate_logistic_model(4096 + i + 1, self.vocab_size)

            current_model.fit_generator(current_generator)

            # save the model to a designated parent folder
            save_path = f'{model_save_directory}/decoder{i+1}'
            current_model.save(save_path)
            
            # clear memory for next model
            reset_keras(current_model)
            
            if verbose:
                print(f'Model #{i+1} saved to {save_path}')
              
        # after training, reload all saved models
        self.load(model_save_directory)
        
                    
    def load(self, directory_path):
        '''
        load in a model from a folder that has all decoders saved into it
        
        do not add a / at the end of the directory path
        '''
        for i in range(self.max_len):
            self.models[i] = load_model(f'{directory_path}/decoder{i+1}')
            
        print(f'Model loaded from {directory_path}')
        
                
    def generate_caption(self, image_filename, verbose=True):
        '''
        given a filename use the trained models to decode each next word for a full caption
        '''
        np.random.seed(RANDOM_SEED)
        
        caption = [START_TOK]

        image_vector = all_image_features[image_filename].reshape(-1,)

        for i in range(self.max_len):
            # should be length i + 1 because one word is added each iteration
            caption_as_indices = self.tokenizer.texts_to_sequences([caption])[0] 
            
            # should be length 4096 + (i+1) for the input of the corresponding decoder
            next_input = np.concatenate([image_vector, np.array(caption_as_indices)])
            
            # reshape it into 1 x 4096 + (i+1) shape for keras input
            next_input = next_input.reshape(1, len(next_input))

            # get the current model
            current_model = self.models[i]
            
            # get the probability distribution for output layer
            probablities = current_model.predict(next_input).reshape(-1,)
                    
            # predict the index of the next word (randomly sample from the vocab based on the prediction output distribution)
            predicted_word_index = np.random.choice(self.vocab_size, p=probablities)
            
            # convert the index to a word based on the tokenizer
            predicted_word = self.tokenizer.index_word[predicted_word_index]

            # if it is the end of sequence token break out the loop
            if predicted_word == END_TOK:
                break
        
            # add the word to out caption (dont add end)
            caption.append(predicted_word)
            
        if verbose:
            print(f'Caption for {image_filename}: {caption}')

        # remove start token
        return caption[1:]