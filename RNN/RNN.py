# Textual Data - Sentiment Analysis
# IMDB has sarcastic reviews

"""
This program will classify reviews from IMDB based on sentiment, positive or
negative.  We will use the IMDB database that comes with Keras.
"""
"""
This data has already preprocessed the reviews. This preprocessing replaces the
actual works with the encoding.  So, the second most popular word is replaced by 2,
third most popular by 3, etc.
"""

# Handles sequences of data
from keras.preprocessing import sequence

from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM
from keras.callbacks import EarlyStopping
from keras.datasets import imdb

# Suppress warning and informational messages
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = '3'

NUM_WORDS = 6000 # the top most n frequent words to consider
SKIP_TOP = 0 # Skip the top most words that are likely (the, and, a)
MAX_REVIEW_LEN = 400 # Max number of words from a review

# Load preprocessed sentiment classified review data from IMDB database
(x_train, y_train), (x_test, y_test) = imdb.load_data(
    num_words=NUM_WORDS, skip_top=SKIP_TOP
)

# Print a sample
# print("encoded word sequence:", x_train[3])

x_train = sequence.pad_sequences(x_train, maxlen=MAX_REVIEW_LEN)
x_test = sequence.pad_sequences(x_test, maxlen=MAX_REVIEW_LEN)

print('x_train.shape:', x_train.shape, 'x_test.shape', x_test.shape)

model = Sequential()

model.add(Embedding(NUM_WORDS, 64))
model.add(LSTM(128))
model.add(Dense(1, activation='sigmoid'))

# Compile
model.compile(
    loss='binary_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)

# Train
BATCH_SIZE = 24
EPOCHS = 5

cbk_early_stopping = EarlyStopping(monitor='val_acc', mode='max')

model.fit(
    x_train, y_train, BATCH_SIZE, epochs=EPOCHS, validation_data=(x_test, y_test),
    callbacks=[cbk_early_stopping]
)
