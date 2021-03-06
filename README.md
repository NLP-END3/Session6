# Session6
The School of AI - END3 Course, Session 6 Assignment

*Group Members: Dhruba Adhikary, Phani Nandula, Prateek Maheshwari, Sujit Ojha*

## 1. Assignment (300 points):
- Train model we wrote in the class on the following two datasets taken from this [link](https://kili-technology.com/blog/chatbot-training-datasets/): 
    - http://www.cs.cmu.edu/~ark/QA-data/ 
    - https://quoradata.quora.com/First-Quora-Dataset-Release-Question-Pairs
- Once done, please upload the file to GitHub and proceed to share the things asked below:
    - Share the link to your GitHub repo (100 pts for code quality/file structure/model accuracy) (100 pts)
    - Share the link to your readme file (100 points for proper readme file)
    - Copy-paste the code related to your dataset preparation for both datasets.  (100 pts)
    - If your model trains and gets to respectable accuracy (200 pts).   

Please remember that the objective of this assignment is to learn how to write code step by steps, so I should be seeing your exploration steps.

## Results Summary

- Question Answer Dataset [Notebook](Q&A_Implementation_using_Seq2Seq_and_attention.ipynb) : Training Accuracy = 90% (9/10 Questions answered correctly)
- Quora Dataset [Notebook](Quora-similarity.ipynb) : Training Accuracy = 70%

## Data Preparation - Question Answer
```Python
def readFile(input_name, output_name):
    pairs = []

    # Read the file and split into lines
    base_dir = '/content/Question_Answer_Dataset_v1.2/'
    folder_names = ['S08/', 'S09/', 'S10/']
    file_name = 'question_answer_pairs.txt'
    
    for folder_name in folder_names:
      print("Reading lines: ", folder_name)
      lines = open(
        base_dir + folder_name + file_name,
        encoding='utf-8',
        errors='ignore'
      ).read().strip().split('\n')

      # Split every line into pairs and normalize
      pairs.extend(
          [[normalizeString(s) for s in l.split('\t')[1:3]] for l in lines[1:]]
      )

    input_question = Lang(input_name)
    output_answer = Lang(output_name)

    return input_question, output_answer, pairs
```

## Data Preparation - Quora
```Python
df = pd.read_csv("/content/quora_duplicate_questions.tsv", sep='\t')
df = df[df['is_duplicate'] ==1]
df['input'] = df["question1"] + '\t' + df["question2"]
li = df['input'].to_list()

with open('eng-eng.txt', 'w') as f:
  for i in li:
    f.write(i)
    f.write('\n')

def readLangs(lang1, lang2, reverse=False):
    print("Reading lines...")

    # Read the file and split into lines
    lines = open('/content/%s-%s.txt' % (lang1, lang2), encoding='utf-8').\
        read().strip().split('\n')

    # Split every line into pairs and normalize
    pairs = [[normalizeString(s) for s in l.split('\t')] for l in lines]

    # Reverse pairs, make Lang instances
    if reverse:
        pairs = [list(reversed(p)) for p in pairs]
        input_lang = Lang(lang2)
        output_lang = Lang(lang1)
    else:
        input_lang = Lang(lang1)
        output_lang = Lang(lang2)

    return input_lang, output_lang, pairs

```

## QUESTION-ANSWER DATASET
Manually-generated factoid question/answer pairs with difficulty ratings from Wikipedia articles. Dataset includes articles, questions, and answers.

- There are three directories, one for each year of students: S08, S09, and S10.
- The file "question\_answer\_pairs.txt" contains the questions and answers.
- The first line of the file contains column names for the tab-separated data fields in the file.
- ArticleTitle | Question | Answer | DifficultyFromQuestioner | DifficultyFromAnswerer | ArticleFile

**Data Fields:**

- Field 1 is the name of the Wikipedia article from which questions and answers initially came.
- Field 2 is the question.
- Field 3 is the answer.
- Field 4 is the prescribed difficulty rating for the question as given to the question-writer.
- Field 6 is the relative path to the prefix of the article files. html files (.htm) and cleaned text (.txt) files are provided.

**Data Samples:**

- Abraham\_Lincoln Who suggested Lincoln grow a beard? Grace Bedell. hard medium data/set3/a4

## What???s a Seq2Seq Model?

A Seq2Seq model is a model that takes a sequence of items (words, letters, time series, etc) and outputs another sequence of items.

![](https://github.com/NLP-END3/Session6/blob/main/Images/Aspose.Words.8686f41a-49b8-4e06-8996-2e77be4aa837.001.png)

Seq2Seq Model

In the case of Neural Machine Translation, the input is a series of words, and the output is the translated series of words.

Now let's work on reducing the blackness of our black box. The model is composed of an??*encoder*??and a??*decoder*. The encoder captures the??*context*??of the input sequence in the form of a??*hidden state vector*??and sends it to the decoder, which then produces the output sequence. Since the task is sequence based, both the encoder and decoder tend to use some form of RNNs, LSTMs, GRUs, etc. The hidden state vector can be of any size, though in most cases, it's taken as a??[power of 2](https://datascience.stackexchange.com/questions/16416/why-the-number-of-neurons-or-convolutions-chosen-equal-powers-of-two)??and a large number (256, 512, 1024) which can in some way represent the complexity of the complete sequence as well as the domain.

![](https://github.com/NLP-END3/Session6/blob/main/Images/Aspose.Words.8686f41a-49b8-4e06-8996-2e77be4aa837.002.png)

Let???s go Deeper! RNNs!

![](https://github.com/NLP-END3/Session6/blob/main/Images/Aspose.Words.8686f41a-49b8-4e06-8996-2e77be4aa837.003.png)

RNN Cell

RNNs by design, take two inputs, the current example they see, and a representation of the previous input. Thus, the output at time step??*t*??depends on the current input as well as the input at time??*t-1*. This is the reason they perform better when posed with sequence related tasks. The sequential information is preserved in a hidden state of the network and used in the next instance.

The Encoder, consisting of RNNs, takes the sequence as an input and generates a final embedding at the end of the sequence. This is then sent to the Decoder, which then uses it to predict a sequence, and after every successive prediction, it uses the previous hidden state to predict the next instance of the sequence.

![](https://github.com/NLP-END3/Session6/blob/main/Images/Aspose.Words.8686f41a-49b8-4e06-8996-2e77be4aa837.004.png)

Encoder-Decoder Model for Seq2Seq Modelling

***Drawback:***??The output sequence relies heavily on the context defined by the hidden state in the final output of the encoder, making it challenging for the model to deal with long sentences. In the case of long sequences, there is a high probability that the initial context has been lost by the end of the sequence.

***Solution:??[***Bahdanau et al., 2014](https://arxiv.org/abs/1409.0473)***??and??[Luong et al., 2015](https://arxiv.org/abs/1508.04025)??papers introduced and a technique called ???Attention??? which allows the model to focus on different parts of the input sequence at every stage of the output sequence allowing the context to be preserved from beginning to end.

Now I???m getting your ATTENTION! ;P

To put it in very simple terms, since the issue was that a single hidden state vector at the end of the encoder wasn???t enough, we send as many hidden state vectors as the number of instances in the input sequence. So here is the new representation:

![](https://github.com/NLP-END3/Session6/blob/main/Images/Aspose.Words.8686f41a-49b8-4e06-8996-2e77be4aa837.005.png)

Seq2Seq with Attention ??? incomplete

Well, that sounds pretty simple, doesn???t it? Let???s bring in some more complexity. How exactly does the Decoder use the set of hidden state vectors? Until now, the only difference between the two models is the introduction of the hidden states of all the instances of the input during the decoding phase.

Another valuable addition to creating the Attention based model is the??*context vector*. This is generated for every time instance in the output sequences. At every step, the context vector is a weighted sum of the input hidden states as given below:

![](https://github.com/NLP-END3/Session6/blob/main/Images/Aspose.Words.8686f41a-49b8-4e06-8996-2e77be4aa837.006.png)

Context Vector

But how is the context vector used in the prediction? And how are the weights??*a1, a2, a3*??decided? Let's go one question at a time, simpler one first ??? the context vector.

The generated context vector is combined with the hidden state vector by concatenation and this new??*attention hidden vector*??is used for predicting the output at that time instance. Note that this attention vector is generated for every time instance in the output sequence and now replaces the hidden state vector.

![](https://github.com/NLP-END3/Session6/blob/main/Images/Aspose.Words.8686f41a-49b8-4e06-8996-2e77be4aa837.007.png)

Attention hidden state

Now we get to the final piece of the puzzle, the attention scores.

Again, in simple terms, these are the output of another neural network model, the??*alignment model*, which is trained jointly with the seq2seq model initially. The alignment model scores how well an input (represented by its hidden state) matches with the previous output (represented by attention hidden state) and does this matching for every input with the previous output. Then a softmax is taken over all these scores and the resulting number is the attention score for each input.

![](https://github.com/NLP-END3/Session6/blob/main/Images/Aspose.Words.8686f41a-49b8-4e06-8996-2e77be4aa837.008.png)

Attention scoring

Hence, we now know which part of the input is most important for the prediction of each of the instances in the output sequence. In the training phase, the model has learned how to align various instances from the output sequence to the input sequence. Below is an illustrated example of a machine translation model, shown in a matrix form. Note that each of the entries in the matrix is the attention score associated with the input and the output sequence.

![](https://github.com/NLP-END3/Session6/blob/main/Images\Aspose.Words.8686f41a-49b8-4e06-8996-2e77be4aa837.009.png)

French to English conversion. Notice how the model weighted the input sequence while outputing European Economic Area???

So now we have the final and complete model

![](https://github.com/NLP-END3/Session6/blob/main/Images/Aspose.Words.8686f41a-49b8-4e06-8996-2e77be4aa837.010.png)

Seq2Seq Attention Based Model

As we can see, the black box that we started with, has now turned white. Below is a pictorial summarization:

![](https://github.com/NLP-END3/Session6/blob/main/Images/Aspose.Words.8686f41a-49b8-4e06-8996-2e77be4aa837.011.png)

![](https://github.com/NLP-END3/Session6/blob/main/Images\Aspose.Words.8686f41a-49b8-4e06-8996-2e77be4aa837.012.png)

