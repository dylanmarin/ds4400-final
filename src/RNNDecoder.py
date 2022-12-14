from common import ALL_FILENAMES, START_TOK, END_TOK
from common import get_tokenizer_from_samples, import_image_features, max_and_average_sequence_length, RANDOM_SEED
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Dropout, Embedding, LSTM, Dense, Input, add
from keras.models import Model
import numpy as np
import tensorflow as tf


all_image_features = import_image_features(
    '../data/flickr_8k/8k_features.pkl', ALL_FILENAMES)


def generate_RNN_no_dropout(vocab_size, max_length, opt='adam'):
    '''
    this function generates an RNN keras model that does not use dropout layers
    vocab_size: is the size of the output layer
    max_length: determines the size of the text input layer. text input will need to be padded if the sequence being given is less than the full length 
    opt: the optimizer to use when compiling the model
    '''
    # first input - VGG generated image features
    image_input = Input(shape=(4096,))
    condensed_image = Dense(256, activation='relu')(image_input)

    # taking in text input which is words 1 through n-1 where y is next word
    # all sequences are padded to be max_length so netowrk has same sized inputs
    text_input = Input(shape=(max_length,))
    embedding_layer = Embedding(vocab_size, 256, mask_zero=True)(text_input)
    lstm_layer = LSTM(256)(embedding_layer)

    # combining condensed image and text layers via addition
    combo_layer1 = add([condensed_image, lstm_layer])
    combo_layer2 = Dense(256, activation='relu')(combo_layer1)

    # softmax layer for all words in vocabulary to generate final prediction
    output = Dense(vocab_size, activation='softmax')(combo_layer2)

    # creating and compiling model
    model = Model(inputs=[image_input, text_input], outputs=output)
    model.compile(loss='categorical_crossentropy', optimizer=opt)
    return model


def generate_RNN_with_dropout(vocab_size, max_length, opt='adam'):
    '''
    this function generates an RNN keras model that includes 2 dropout layers
    vocab_size: is the size of the output layer
    max_length: determines the size of the text input layer. text input will need to be padded if the sequence being given is less than the full length 
    opt: the optimizer to use when compiling the model
    '''
    # first input - VGG generated image features
    image_input = Input(shape=(4096,))

    # 50% droupout
    dropout_1 = Dropout(0.5)(image_input)
    condensed_image = Dense(256, activation='relu')(dropout_1)

    # taking in text input which is words 1 through n-1 where y is next word
    # all sequences are padded to be max_length so netowrk has same sized inputs
    text_input = Input(shape=(max_length,))
    embedding_layer = Embedding(vocab_size, 256, mask_zero=True)(text_input)

    # 50% droupout
    dropout_2 = Dropout(0.5)(embedding_layer)
    lstm_layer = LSTM(256)(dropout_2)

    # combining condensed image and text layers via addition
    combo_layer1 = add([condensed_image, lstm_layer])
    combo_layer2 = Dense(256, activation='relu')(combo_layer1)

    # softmax layer for all words in vocabulary to generate final prediction
    output = Dense(vocab_size, activation='softmax')(combo_layer2)

    # creating and compiling model
    model = Model(inputs=[image_input, text_input], outputs=output)
    model.compile(loss='categorical_crossentropy', optimizer=opt)
    return model


class RNNModel():
    '''
    This model is a wrapper class to define both of our RNN model types
    It can be used to train, save, and load RNN models, both with and without dropout layers
    '''
    def __init__(self, dropout_layer, samples, optimizer='adam'):
        '''
        initialize our RNN model class
        dropout_layer: boolean - whether to include the two dropout layers or not
        samples: the samples that this model will be trained with. it is used to 
            learn the vocabulary for the tokenizer that will convert words to their 
            indices and back
        optimizer: the optimizer that will be used to compile the model with
        '''
        if dropout_layer:
            self.generate_func = generate_RNN_with_dropout
        else:
            self.generate_func = generate_RNN_no_dropout
        self.optimizer = optimizer
        self.tokenizer = get_tokenizer_from_samples(samples)
        self.vocab_size = len(self.tokenizer.word_index) + 1
        self.max_len = max_and_average_sequence_length(samples)[0]
        self.model = self.generate_func(
            self.vocab_size, self.max_len, opt=self.optimizer)

    def create_sequences(self, image_vector, captions):
        '''
        Given:
        - one image vector representing a single image
        - one value from our samples dict for the corresponding image (a list of 5 tokenized captions)

        For each caption: 
            Break it into samples where for i in range N <= len(caption-1):
                X1 = the image vector
                X2 = the first i words of the caption
                y = the i+1 word in the sequence

        The total numebr of samples should equal the total number of "next words" in all 5 captions
        '''
        # list of image features
        X1 = []
        # word inputs (as word indexes)
        X2 = []
        # next word
        y = []

        # reshape the image vector for the model input
        image_vector = image_vector.reshape(-1,)

        # convert the words (strings) descriptions corresponding to this image into numbers using the tokenizer 
        descriptions = self.tokenizer.texts_to_sequences(captions)
        
        # for each caption/description
        for description in descriptions:
            # for each position in the description (except for the last) create a sample
            for i in range(len(description) - 1):
                # where x1 input is the image vector
                X1.append(image_vector)
                
                # x2 is the sequence so far
                # pad the sequence x2 to always be max sequence length
                x2 = pad_sequences([description[:i + 1]],
                                   maxlen=self.max_len, padding='post')[0]
                X2.append(x2)
                
                # y is the next word in the sequence encoded as a 1-hot vector
                # vector is the size of the vocab and the index corresponding to the next word is 1
                y.append(to_categorical(description[i+1], self.vocab_size))

        return tf.convert_to_tensor(np.asarray(X1)), tf.convert_to_tensor(np.asarray(X2)), tf.convert_to_tensor(y)

    
    def data_generator(self, filename_description_dictionary, epochs):
        '''
        create a python generator to be used in training
        filename_description_dictionary: dictoinary of samples where the key is the 
            image filename and the value is the five corresponding captions
        epochs: the number of times to loop the generator (since they are traditionally one use)
        '''
        for _ in range(epochs):
            np.random.seed(RANDOM_SEED)

            # shuffle filename order for better distribution over multiple loops (epochs)
            all_filenames = list(filename_description_dictionary.keys())
            np.random.shuffle(all_filenames)

            # loop for ever over files
            for filename in all_filenames:
                # get the corresponding descriptions
                descriptions = filename_description_dictionary[filename]

                # retrieve the photo feature
                image_feature_vector = all_image_features[filename][0]

                image_input, sequence_input, next_word = self.create_sequences(
                    image_feature_vector, descriptions)
                yield [image_input, sequence_input], next_word

    def train_save_model(self, input_dict, save_path, epochs=1):
        '''
        train and save the model
        input_dict: the dictionary of samples to train from
            key is image filename, value is 5 corresponding captions
        save_path: is the path to save the model to
        epochs: is the number of epochs to iterate through the training data
        
        this saves the weights to an h5 format which is an older format, but 
        solved an issue we were having with loading
        '''
        generator = self.data_generator(input_dict, epochs)
        self.model.fit(generator)
        self.model.save(save_path, save_format='h5')
        return self.model

    def generate_caption(self, image_filename, verbose=True):
        '''
        given a keras model, and an image_filename

        use the model to generate a caption token by token
        
        imports the features for the given image filename (previously extracted)

        randomly samples next token from models output probabilities and uses the 
        whole caption including the next word for the next position
        
        generates until it either reaches the maximum length or generates the <end> token
        '''
        np.random.seed(RANDOM_SEED)

        tokens = [START_TOK]

        # get the features of the image we want to generate a caption for
        file_image_features = all_image_features[image_filename]

        # for each position in the caption
        for i in range(0, self.max_len):
            # get the sequence so far as indices (using our tokenizer)
            seq = self.tokenizer.texts_to_sequences([tokens])[0]

            # pad the sequence for our model
            seq = pad_sequences([seq], maxlen=self.max_len, padding='post')

            # get probability distribution for next word based on seq so far
            probabilities = self.model.predict(
                [file_image_features, seq], verbose=0).reshape(-1,)

            # sample the next word based on model output
            pred = np.random.choice(a=self.vocab_size, p=probabilities)

            # convert to word from index of word
            next_word = self.tokenizer.index_word[pred]

            # break out the loop if we generate the end of seq token
            if next_word == END_TOK:
                break

            # otherwise append the next word to the caption so far
            tokens.append(next_word)

        # remove start token (end is never appended)
        tokens = tokens[1:]
        
        if verbose:
            print(f'caption generated for {image_filename}: {tokens}')
        
        return tokens

    def generate_captions_for_files(self, filenames, verbose=True):
        '''
        given a list of filenames, output a list with one generated caption for each filename
        
        uses the generate_caption function to generate a captions for each filename
        '''
        output = []
        for filename in filenames:
            output.append(self.generate_caption(filename, verbose))
        return output

    def load(self, filepath):
        '''
        given the filepath that a model was previously saved to,
        load the weights into this model
        '''
        self.model.load_weights(filepath)
        print('model loaded successfully!')
        return