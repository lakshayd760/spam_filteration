# spam_filteration
* This project is for classification of spam emails from the ham emails
* For this project we have used neural networks and Bert pretrained model

## Techstack
* Pandas
* Tensorflow
* sklearn
* Bert

## Dataset :
We have used spam email dataset provided by kaggle which you can download using the official kaggle website(https://www.kaggle.com/datasets/shantanudhakadd/email-spam-detection-dataset-classification)<br/>
you can load the dataset using the following code:
>df = pd.read_csv('spam.csv')

Overview of the dataset:
>df.head()

## Preprocessing the data
* Firstly we need to check for null values if exists
  >data.isnull().sum()<br/>
  
* converting the 'spam' and 'ham' labels to numbers using one-hot encoding
  > dataset['labels'] = dataset['tags'].map({'spam':1, 'ham':0})<br/>

## Model building
* Spliting the dataset in training and test set as:
  >from sklearn.model_selection import train_test_split<br/>
  >X_train, X_test, y_train,y_test = train_test_split(dataset['text'], dataset['labels'], test_size=0.2)<br/>
* importing bert_preprocessor and bert_encoder (for preprocessing we can also use tokenizer by nltk or spacy followed by bert vectorization):
  >bert_preprocessor = hub.KerasLayer('https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3')<br/>
  >bert_encoder = hub.KerasLayer('https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/4')<br/>
* forming a model using different layers as:
>input_text = tf.keras.layers.Input(shape=(), dtype=tf.string)<br/>
>preprocess = bert_preprocess(input_text)<br/>
>outputs = bert_model(preprocess)<br/>

>l = tf.keras.layers.Dropout(0.2)(outputs['pooled_output'])<br/>
>l = tf.keras.layers.Dense(1, activation='sigmoid')(l)<br/>
>model = tf.keras.Model(inputs = [input_text], outputs=[l])<br/>
>model.summary()<br/>
* compiling and fiting the model over the training dataset
  >model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])<br/>
  >model.fit(X_train, y_train, epochs=5)<br/>
* evaluation of model on our test set
  >model.evaluate(X_test, y_test)<br/>
our model gives us 95.6% accuracy over the test set
