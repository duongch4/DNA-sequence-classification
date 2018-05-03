## BINF6210, Final Project, Fall 2016 ##
## DUONG, BANG CHI          0981462   ##
########################################

# Import modules
from Bio import SeqIO
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics.pairwise import linear_kernel
import numpy as np
import re
import random

############################### FUNCTIONS ####################################
# Define a function to make a list of kmer - one k only
def kmerList(sequence, k):
    if k >= len(sequence):
        print ("You specified a kmer size of {0} that is"
               "greater or equal to the length {1} of the sequence".format(k, len(sequence)))
        return
    kmers = []
    for word in range(len(sequence)+1-k):
        kmers.append(sequence[word:word+k])
    return kmers

# Define a function to make a list of kmer - many k's
# this is built upon the function kmerList
def kmerListExtend(sequence, k_max):
    #myKmerDict = dict()
    #keys = list(range(2, k_max+1))
    #for index in range(len(keys)):
    myList = []
    for k in range(1, k_max +1):
        myList += kmerList(sequence, k)
    return myList

# Define a function to make a list of repeats (also the number of their occurences -1),
# with a choice of minimum number of letters,
# and a choice of minimum number of repeats (should be at least 2)
def repeatList(sequence, minWord, minRepeat):
    if minRepeat < 2:
        print ("The number of repeats should be at least 2!")
        return
    myRegex = "(?=(.{{{},}})(?:\\1){{{},}})".format(minWord, minRepeat-1)
    repeat = re.compile(myRegex)
    return (repeat.findall(sequence))

############################### CONSTANTS ####################################
k_max = 3
minWord = 2
minRepeat = 2

############################### LOAD DATA ####################################
# Read in data
Files = ["synd1.fasta","synd2.fasta","synd3.fasta","synd4.fasta", "TestData.fasta"]
RecordDictList = []
# Each file will be stored as a dictionary with (key : value) as (SeqID : Seq)
for file in Files:
    RecordDictList.append(SeqIO.index(file, "fasta"))

### Create an array to store 4 Types of Sequences + 1 TestData
### Create an array to store ID's for the five files
# So Seqs[0/1/2/3] will have 50 sequences, and Seqs[4] will have 200 sequences 
Seqs = [[],[],[],[],[]]
SeqsID = [[],[],[],[],[]]

for iFile in range(len(RecordDictList)):
    keys = list(RecordDictList[iFile].keys())
    random.shuffle(keys)
    for key in keys:
        Seqs[iFile].append(str((RecordDictList[iFile])[key].seq))
        SeqsID[iFile].append(key)

### FEATURE SELECTION (choose either "KMERS" or "REPEATS") for each sequence #########
FeatureList = [[],[],[],[],[]]                                                       #
                                                                                     #
for iFile in range(len(Seqs)):                                                       #
    for jSeq in range(len(Seqs[iFile])):                                             #
                                                                                     #
        ## Choose KMERS as feature                                                   #
        FeatureList[iFile].append(kmerListExtend(Seqs[iFile][jSeq], k_max))          
                                                                                     #
        ## Choose REPEATS as feature                                                 #
        #FeatureList[iFile].append(repeatList(Seqs[iFile][jSeq], minWord, minRepeat))
                                                                                     #
######################################################################################

### Make sentences by concatenating feature elements
FeatureToSentence = [[],[],[],[],[]]

for iFile in range(len(FeatureList)):
    for jSeq in range(len(FeatureList[iFile])):
        FeatureToSentence[iFile].append(" ".join(FeatureList[iFile][jSeq]))

### Make a train list (40 sequences) and a validation list (10 sequences) for each Type
### I already randomised the order of keys in each dictionary (line 68),
# so it is safe to say sequences are selected randomly when splitting data into training set and validation set
TrainData, ValidData, ValidDataID = [],[],[]
for iFile in range(len(FeatureToSentence)-1):
    TrainData.append(FeatureToSentence[iFile][0:40])
    ValidData.append(FeatureToSentence[iFile][40:50])
    ValidDataID.append(SeqsID[iFile][40:50])

# Collapse the list of lists into a single list for easy manipulation
ValidDataFlattened = [val for sublist in ValidData for val in sublist]
ValidDataID = [val for sublist in ValidDataID for val in sublist]
# Extract the TestData
TestData = FeatureToSentence[4]

#################### VECTOR SPACE MODEL - TF-IDF #############################
# Declare lists to store information on the analysis
trainVector = [None]*4

trainTermFreqMatrix, validTermFreqMatrix = [None]*4, [None]*4
testTermFreqMatrix = [None]*4

trainIDF = [None]*4

trainTFIDF, validTFIDF = [None]*4, [None]*4
testTFIDF = [None]*4

ValidCosine = [None]*4
TestCosine = [None]*4

# For each Type of sequences
for iType in range(len(trainVector)):
    # Training step (vectorise the training data and fit a model)
    trainVector[iType] = CountVectorizer()
    trainVector[iType].fit_transform(TrainData[iType])
    # Make Term Frequency (TF) Matrices for training set, validation set, and the TestData
    # TF matrices are made based on the trainVector of the previous step
    trainTermFreqMatrix[iType] = trainVector[iType].transform(TrainData[iType])
    validTermFreqMatrix[iType] = trainVector[iType].transform(ValidDataFlattened)
    testTermFreqMatrix[iType]  = trainVector[iType].transform(TestData)
    # Make Inverse Document Frequency (IDF) based on trainTermFreqMatrix of the previous step
    trainIDF[iType] = TfidfTransformer(norm="l2")
    trainIDF[iType].fit(trainTermFreqMatrix[iType])
    # Make TF-IDF matrices for training set, validation set, and the TestData,
    # They will be normalised based on Euclidean distance metric (L2)
    # Normalised TF-IDF matrices are made based on the trainIDF of the previous step
    trainTFIDF[iType] = trainIDF[iType].transform(trainTermFreqMatrix[iType])
    validTFIDF[iType] = trainIDF[iType].transform(validTermFreqMatrix[iType])
    testTFIDF[iType]  = trainIDF[iType].transform(testTermFreqMatrix[iType]) 
    # Calculate the cosine of the angle between vectors of training set vs validation set,
    # and between vectors of training set vs the TestData
    # These are matrices => [Rows : Cols] as [Train Seqs : Valid/Test Seqs], and each cell contains the cosine value
    ValidCosine[iType] = linear_kernel(trainTFIDF[iType], validTFIDF[iType])
    TestCosine[iType]  = linear_kernel(trainTFIDF[iType], testTFIDF[iType])

# Find the maxima of cosine (most similar) across TrainData Sequences
# (across 40 rows, axis=0, of the Cosine matrices)
# Cols are Sequences of ValidData (10 cols) or TestData (40 cols)
# Store the maxima in a list. This is a list of 4 sublists corresponding to 4 types.
MaxCosAcrossRowsValid = []
MaxCosAcrossRowsTest = []

for iType in range(len(TestCosine)):
    MaxCosAcrossRowsValid.append(ValidCosine[iType].max(axis=0))
    MaxCosAcrossRowsTest.append(TestCosine[iType].max(axis=0))

# Turn the two lists into two 2-dimensional matrices,
# with [Rows : Cols] as [4 Types : (10 or 200) Maxima]
MaxCosAcrossRowsValid = np.array(MaxCosAcrossRowsValid)
MaxCosAcrossRowsTest = np.array(MaxCosAcrossRowsTest)

# Find the maximum of cosine across Types and compare it to the maximum of ValidCosine/TestCosine
# => determine which Type is the most similar to the Valid/TestData Seq
# (across 4 rows, axis=0, of MaxCos matrices)
# Cols are Maxima of cosine between (ValidSeq(10 cols) or TestDataSeq(200 cols)) and (the 40 TrainSeq)
# Store the new maxima in a list. This is a list of 4 sublists corresponding to 4 types.
ValidResult = [[],[],[],[]]
TestDataResult = [[],[],[],[]]

# For each column in the MaxCosAcrossRowsValid matrix
for iCol in range(np.shape(MaxCosAcrossRowsValid)[1]):
    # If the maximum of MaxCos matrix equals the maximum of ValidCosine for Type 1
    if ( (MaxCosAcrossRowsValid.max(axis=0))[iCol] == (ValidCosine[0].max(axis=0))[iCol] ):
        # Store the sequence ID to the ValidResult for Type 1. Carry on with the if-statement
        ValidResult[0].append(ValidDataID[iCol])
    elif ( (MaxCosAcrossRowsValid.max(axis=0))[iCol] == (ValidCosine[1].max(axis=0))[iCol] ):
        ValidResult[1].append(ValidDataID[iCol])
    elif ( (MaxCosAcrossRowsValid.max(axis=0))[iCol] == (ValidCosine[2].max(axis=0))[iCol] ):
        ValidResult[2].append(ValidDataID[iCol])
    else: 
        ValidResult[3].append(ValidDataID[iCol])

# For each column in the MaxCosAcrossRowsTest matrix, do the same as we did with Validation set
for iCol in range(np.shape(MaxCosAcrossRowsTest)[1]):
    if ( (MaxCosAcrossRowsTest.max(axis=0))[iCol] == (TestCosine[0].max(axis=0))[iCol] ):
        TestDataResult[0].append(SeqsID[4][iCol])
    elif ( (MaxCosAcrossRowsTest.max(axis=0))[iCol] == (TestCosine[1].max(axis=0))[iCol] ):
        TestDataResult[1].append(SeqsID[4][iCol])
    elif ( (MaxCosAcrossRowsTest.max(axis=0))[iCol] == (TestCosine[2].max(axis=0))[iCol] ):
        TestDataResult[2].append(SeqsID[4][iCol])
    else: 
        TestDataResult[3].append(SeqsID[4][iCol])

############################ PRINT THE RESULTS ###############################
print ("======== TRUE vs PREDICTION VALUES ON CROSS-VALIDATION DATA ==========")

for iType in range(4):
    print ("True Type {0}:\n {1}".format(iType +1, np.sort(SeqsID[iType][40:50])))

print ("======================================================================")

for iType in range(4):
    print ("There are {0} Predicted Type {1}:\n {2}".format(len(ValidResult[iType]), iType +1, np.sort(ValidResult[iType])))

print ("\n\n============= PREDICTION ON 200 SEQUENCES TEST DATA ==================")

for iType in range(4):
    print ("\nThere are {0} Type {1} Sequences:\n {2}".format(len(TestDataResult[iType]), iType +1, np.sort(TestDataResult[iType])))

################################# THE END #####################################
