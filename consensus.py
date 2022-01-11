from pathlib import Path
import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from scipy.spatial import distance
import re
import chardet


def lowercase(x):
    '''Lowercase the text.
    '''
    x = x.lower()
    return x


def remove_punct(x):
    '''Concatenate contractions into one word, then remove all 
    non-alphanumeric, non-space characters
    '''
    # Morph contractions into one word rather than splitting
    step1 = re.sub("['’]", '', x)
    # Remove other non alphanumerics, replacing them with spaces
    step2 = re.sub('[^A-Za-z\s]', ' ', step1)
    return step2


def remove_whitespace(x):
    ''' Remove extra newlines, lines of only space, and big spaces, replacing
    them with single spaces.'''
    # Remove tabs
    step1 = re.sub('\t', '', x)
    # Remove lines containing only whitespace
    step2 = re.sub(r'\n\s*\n', '\n', step1)
    # Replace multiple spaces with single spaces
    done = re.sub(' +', ' ', step2)
    return done


def freq_dist(inlist):
    '''Turn list of words (input) into a frequency distribution.
    '''
    df = pd.Series(inlist).value_counts()
    return df


def cleantextname(bookname):
    ''' Pull the .txt portion off of the filename.
    '''
    # Regex for detecting the name
    regex = '(.+)(?:\.txt)'
    # Get the bookname, and return it.
    x = re.search(regex, bookname)
    return x.group(1)


def preclean(x):
    '''Perform cleaning operations in order to lowercase, remove punctuation, 
    and space from input and return clean text.
    '''
    x = lowercase(x)
    x = remove_punct(x)
    x = remove_whitespace(x)
    return x


def make_dists(in_path, out_path):
    '''Create word frequency distributions for each text file in corpus.
    in_path: string location of text files
    out_path: string location where frequency distribution csvs will be written
    '''
    # Create dictionary for temporary storage
    storage_dict = {}
    # Change input location to Path object
    data_folder = Path(in_path)
    # Get list of files in data folder
    for filename in os.listdir(data_folder):
        # Check if the file is a text file
        if filename.endswith('.txt'):
            # Open the file to detect encoding
            with open(os.path.join(data_folder, filename), 'rb')as my_file:
                # Try to read file
                text = my_file.read()
                # Detect encoding
                result = chardet.detect(text)
                # Get encoding
                charenc = result['encoding']
            # Open the file in read mode, using the proper encoding
            with open(os.path.join(data_folder, filename), 'r', encoding = charenc) as my_file:
                # Read text, assign to a variable
                text = my_file.read()
                # Put text variable in dictionary for storage
                # Each text is associated with its filename (usually document title)
                storage_dict[filename] = text
    # Iterate over storage dictionary
    for book, text in storage_dict.items():
        # Clean the text
        clean = preclean(text)
        # Convert text into a frequency distribution
        df = freq_dist(clean.split())
        # Create outfilepath including document name
        out = Path(str(out_path) + cleantextname(book)+'_frequency_distribution.csv')
        # Write to CSV
        df.to_csv(out)
    return


def clean_dist_name(bookname):
    ''' Pull the excess portion off of the frequency distribution filename.
    '''
    # Regular expression to search for the right portion of the filename
    regex = '(.+)(?:_frequency_distribution\.csv)'
    # Perform search
    x = re.search(regex, bookname)
    return x.group(1)


def calculate_relative_frequencies(data):
    ''' Convert list of raw frequencies into relative frequencies.
    '''
    # Create list to hold relative frequencies
    relative_freq = []
    # Sum the data to get the total
    total = sum(data)
    # Perform item/total division to get relative frequency
    for item in data:
        relative_freq.append(item/total)
    return relative_freq


def relative_frequencies(in_frame):
    ''' Turn 'count' column from DataFrame into relative counts. Takes 
    DataFrame as input and returns it, with counts relativized
    '''
    # Pop out count column
    column = in_frame.pop('Count')
    # Relativize frequencies in the target column
    freqs = calculate_relative_frequencies(column.tolist())
    # Turn frequencies into array so we can add them back to the dataframe.
    freqs = np.asarray(freqs)
    # Add Count column back in with relative frequencies.
    in_frame['Count'] = freqs
    return in_frame


def manhattan_distance(vectors, size):
    '''Calculate the Manhattan (cityblock) distance between two many-dimensional vectors
    Returns array of distances between vectors in an (x,y) coordinate system 
    (each axis corresponds to a list of documents). 
    The value at their x,y intersection will be the distance between doc x and doc y.
    '''
    # Create empty array of correct size
    tempdistances = np.empty((size, size))
    # Iterate over each x,y pair
    for row_value in range(size):
        for column_value in range(size):
            # Calculate 
            if row_value != column_value:
                tempdistances[row_value][column_value] = distance.cityblock(vectors[row_value], vectors[column_value])
            # If x & y are equal, the distance between them is zero
            else:
                tempdistances[row_value][column_value] = 0
    return tempdistances


def euclidean_distance(vectors, size):
    '''Calculate the Euclidean distance between two many-dimensional vectors
    Returns array of distances between vectors in an (x,y) coordinate system 
    (each axis corresponds to a list of documents). 
    The value at their x,y intersection will be the distance between doc x and doc y.
    '''
    # Create empty array of correct size
    tempdistances = np.empty((size, size))
    # Iterate over each x,y pair
    for row_value in range(size):
        for column_value in range(size):
            # Calculate 
            if row_value != column_value:
                tempdistances[row_value][column_value] = distance.euclidean(vectors[row_value], vectors[column_value])
            # If x & y are equal, the distance between them is zero
            else:
                tempdistances[row_value][column_value] = 0
    return tempdistances    
    

def cosine_distance(vectors, size):
    '''Calculate the Cosine distance between two many-dimensional vectors
    Returns array of distances between vectors in an (x,y) coordinate system 
    (each axis corresponds to a list of documents). 
    The value at their x,y intersection will be the distance between doc x and doc y.
    '''
    # Create empty array of correct size
    tempdistances = np.empty((size, size))
    # Iterate over each x,y pair
    for row_value in range(size):
        for column_value in range(size):
            # Calculate 
            if row_value != column_value:
                tempdistances[row_value][column_value] = distance.cosine(vectors[row_value], vectors[column_value])
            # If x & y are equal, the distance between them is zero
            else:
                tempdistances[row_value][column_value] = 0
    return tempdistances  


def minkowski_distance(vectors, size):
    '''Calculate the Minkowski distance between two many-dimensional vectors
    Returns array of distances between vectors in an (x,y) coordinate system 
    (each axis corresponds to a list of documents). 
    The value at their x,y intersection will be the distance between doc x and doc y.
    '''
    # Create empty array of correct size
    tempdistances = np.empty((size, size))
    # Iterate over each x,y pair
    for row_value in range(size):
        for column_value in range(size):
            # Calculate 
            if row_value != column_value:
                tempdistances[row_value][column_value] = distance.minkowski(vectors[row_value], vectors[column_value])
            # If x & y are equal, the distance between them is zero
            else:
                tempdistances[row_value][column_value] = 0
    return tempdistances    


def correlation_distance(vectors, size):
    '''Calculate the Correlation distance between two many-dimensional vectors
    Returns array of distances between vectors in an (x,y) coordinate system 
    (each axis corresponds to a list of documents). 
    The value at their x,y intersection will be the distance between doc x and doc y.
    '''
    # Create empty array of correct size
    tempdistances = np.empty((size, size))
    # Iterate over each x,y pair
    for row_value in range(size):
        for column_value in range(size):
            # Calculate 
            if row_value != column_value:
                tempdistances[row_value][column_value] = distance.correlation(vectors[row_value], vectors[column_value])
            # If x & y are equal, the distance between them is zero
            else:
                tempdistances[row_value][column_value] = 0
    return tempdistances      
 

def create_rankings(distances, forces, n = 3):
    ''' Given an array of the distances between one node and each other, choose the closest n connections.
    Assign the closest a score of n, and the second n-1, until the nth closest with a score of 1.
    Takes the following variables as parameters:
    n: The number of runners-up to have (including first place). 3 works best.
    distances: an array representing the calculated distance between a given document and each other document in the corpus
    forces: An array representing the total strength of all connections, we pass it in, add to it, and pass it out after updating it
    '''
    # Set array size to size of distance array
    size = len(distances)
    # Iterate over each row in the distances array
    for num in range(size):
        # Get the relevant row from distances array
        row = distances[num]
        # Create empty list to hold values
        values = []
        # Iterate over each value in the row 
        for value in row:
            #If value is 0, there's no comparison to make, so we skip this
            if value != 0:
                # Append float of value to values list
                values.append(float(value))
                # Sort list so values are in order
                values.sort()
        # Do this n times (usually top 3)        
        for i in range(n):
            # Create variable for rank
            rank = n - i
            # Find out where the value is located. This returns an array 
            x = np.where(row == values[i])
            # Assign x the value in the array at x instead of being an array holding that value.
            x = x[0]
            # Check if len(x) is 1. This should always be the case. If not, your corpus has duplicate documents.
            if len(x) == 1:
                # Get x, y coordinates of value
                x = int(x[0])
                y = int(num)
                # Get old value at those coordinates
                old_value = forces[x][y]
                # Add rank to old value at coordinates
                new_value = old_value + rank
                # Assign new value to forces array
                forces[x][y] = new_value
            # Deal with when x contains more than one value (this only happens with duplicates in your corpus, and is a problem)
            else:
                for n in range(len(x)):
                    print("We've run into some issues.")
                    # Let a tie happen (this prevents errors, but is not ideal)
                    a = int(x[n])
                    b = int(num)
                    print(a)
                    print(b)
                    old_value = forces[a][b]
                    new_value = old_value + rank
                    forces[a][b] = new_value           
    # Return forces array, updated                
    return forces


def create_vocab(list_of_freq_dists):
    '''Create vocabulary of entire corpus, sorted on word frequency.
    Takes as input a list of frequeny distributions, 1 for each text
    returns list of Words, sorted in descending order.
    '''
    # Create dictionary for storage
    final_dict = {}
    # Iterate over the list of freq dists
    for text in list_of_freq_dists:
        # Turn frequency distributions into word, value tuples
        tuples = list(text.itertuples(index=False, name=None))
        # Iterate over each tuple in the frequency distribution
        for tup in tuples:
            # Assign word and value to variables
            word = tup[0]
            value = tup[1]
            # Check if the word is in the final dictionary
            if word in final_dict:
                # If word in the dict, add to the old value
                start_value = final_dict[word]
                final_value = start_value + value
                final_dict[word] = final_value
            # Otherwise, add the word to the dictionary
            else:
                final_dict[word] = value
    # Reverse sort the dictionary of all words, and assign to new variable            
    out_dict = sorted(final_dict.items(), key=lambda x: x[1], reverse=True)
    # Make out_dict into a DataFrame
    df = pd.DataFrame(out_dict, columns=['Word', 'Count'])  
    # Pop off the words, and turn them into a list
    words = list(df.pop('Word'))
    # Return the list of words
    return(words)
    

def create_different_length_docs(in_words, vocab_size=1000, low=100, high=1100, step=100):
    '''Truncate the vocabulary into slices of n words.
    Default is 10 slices from 100-1000 
    Returns list of slices 
    '''
    # Create vocabulary, and truncate off words after vocab_size 
    out_list = create_vocab(in_words)[:vocab_size]
    with open ("out.txt", 'w', encoding ="utf-8") as my_file:
        for word in out_list:
            print(word, file=my_file)
    # Split vocab into chunks, adding a certain munber of words each time
    docs = []
    for i in range(low, high, step):
        docs.append(out_list[:i])

    '''  out_list = create_vocab(in_words)
    numbers = [58, 130, 213, 305, 405, 512, 629, 753, 888, 1033]
    docs = []
    for number in numbers:
        docs.append(out_list[:number])'''

    # Return list of vocab, sliced into 10 different lengths
    return docs


def measure_similarity(docs, s_strengths, measures, dfs, n):
    '''Measures the similarity pairwise between each pair of documents.
    Parameters:
    docs: Set of all documents
    measures: list of similarity measures to apply
    s_strengths: array of all the possible similarities, passed in, updated, and passsed out
    dfs: dataframes of word frequencies
    n: number of runners-up (inclusive)
    '''
    # Create counter
    counter = 0
    # create dataframes of each wordlist
    docnum = len(docs)
    for doc in docs:
        counter += 1
        # Notify user
        print(f"Working on layer {counter} of {docnum}...")
        df = pd.DataFrame(index=doc)
        # Iterate over key, value pairs
        for key, value in dfs.items():
            # create wordcount vectors for each text
            templist = []
            for word in doc:
                if word not in value['Count']:
                    templist.append(0)
                else:
                    templist.append(value['Count'][word])
            df[key] = templist
        # Scale the data across the corpus    
        scaler = StandardScaler()
        scaled = scaler.fit_transform(df)
        df[:] = scaled
        columns = list(df.columns)
        frequency_vectors = []
        # Add relativized, scaled data to frequency_vectors list
        for column in columns:
            frequency_vectors.append(df[column].tolist())
        # Calculate similarities between documents for each similarity measure
        for measure in measures:
            s_strengths = create_rankings(measure(frequency_vectors, len(s_strengths)), s_strengths,  n)
    return s_strengths


def perform_consensus(freq_dist_folder, out_path = '', n=3):
    '''Run the functions to perform consensus. Takes the following parameters:
    freq_dist_folder: directory of frequency distributions 
    out_path: path where output csv will go
    n: number of runners-up (inclusive) works best at 3.
    '''
    # Create variables to hold things during function run
    dfs = {}
    text_names = []
    texts = []
    data_folder = Path(freq_dist_folder)
    measures = [
                manhattan_distance, 
                euclidean_distance, 
                cosine_distance, 
                minkowski_distance, 
                correlation_distance
                ]
    # Notify user of what's happening
    print('Preparing Files')
    # Open files and read them into different data structures
    for filename in os.listdir(data_folder):
        # Only read csvs
        if filename.endswith('.csv'):
            # Clean name
            name = clean_dist_name(filename)
            # Add names to own variable
            text_names.append(name)
            # Add texts to list
            texts.append(pd.read_csv(Path(freq_dist_folder + filename), header=None))
            # Make relative frequencies into DataFrame
            df = relative_frequencies(pd.read_csv(Path(freq_dist_folder + filename), names=['Count'], index_col=0))
            # Turn df into a dictionary
            dfs[name] = df.to_dict(orient='dict')
    array_size = len(text_names)
    # Determine how many texts there are, and create empty array of textsize x textsize
    similarity_strengths = np.zeros((array_size, array_size), dtype=int)
    # Notify user of what's happening
    print('Creating Wordlists')
    # Create wordlists from corpus
    docs = create_different_length_docs(texts)
    # Notify user of what's happening
    print('Measuring Similarity (This may take a while)')
    # Calculate document similarities
    similarity_strengths = measure_similarity(docs, similarity_strengths, measures, dfs, n )
    # Notify user of what's happening
    print('Creating Output')
    # Create output data            
    out = (similarity_strengths)
    out_data = pd.DataFrame(data=out, index=text_names, columns=text_names)
    # Write out_data to .csv
    out_data.to_csv(Path(out_path + 'out.csv'))
    # Return location of out csv.
    return out_path


def check_filepath(path):
    '''Given a string filepath to frquency distributions, check if it exists. 
    If not, make it. Return Path object where distributions are located
    '''
    # Set relative filepath location
    extra_part = '/frequency_distributions/'
    # Concatenate with input to create full directory.
    out_directory = path + extra_part
    # Make directory into Path object
    freq_dist_location = Path(out_directory)
    # Check if directory exists
    directory_exists = os.path.isdir(freq_dist_location)
    # If true, do nothing    
    if directory_exists == True:
        pass
    # Else, create directory
    else:
        os.mkdir(freq_dist_location)
    # Return the directory as a string
    return out_directory
        

def freqdist(in_filepath):
    '''Given the filepath to a directory of text files, create a word-frequency 
    distribution of each file in that directory
    '''
    # Check if filepath exists, create it if not. Assign path to variable
    freq_dist_location = check_filepath(in_filepath)
    # Create frequency distributions in proper location
    make_dists(in_filepath, freq_dist_location)
    return freq_dist_location


def create_consensus_matrix(freq_dist_location):
    '''Create consensus matrix given Path to frequency distributions
    '''
    freq_dist_location = str(freq_dist_location) +'\\'
    out_path = perform_consensus(freq_dist_location)
    return out_path
    

def create_edge_and_node_files(filepath):
    '''Creates edges.csv and nodes.csv files 
    Input: filepath -- location of out.csv output created earlier
    Also where edges and nodes will be written to
    These edge and node files are standard input for force-directed graph software.
    '''
    # Create Path objects for each file location 
    out_path = Path(filepath + 'out.csv')
    edges_path = Path(filepath + 'edges.csv')
    nodes_path = Path(filepath + 'nodes.csv')
    # Read in data from csv
    data = pd.read_csv(out_path , header=0, index_col=0)
    # Create new array from data read in
    new_array = data.to_numpy()
    # Variables for columns and size
    cols = list(data.columns)
    size = len(cols)
    # Create ids list
    ids = []
    # Fill ids list
    [ids.append(i) for i in range(size)]
    # Zip ids and labels for nodes document    
    nodes = pd.DataFrame(list(zip(ids, cols)), columns=['Id', 'Label'])
    # Export node document to csv
    nodes.to_csv(nodes_path, index=False)   
    # Create edges variable
    edges = []
    # Iterate through whole array
    for x in ids:
        for y in ids:
            # Add non-zero values to edge totals
            if new_array[x][y] != 0:
                edges.append((y, x, new_array[x][y]))
    # Create DataFrame object of edge source, target, and weight variables
    edge_df = pd.DataFrame(edges, columns=['Source', 'Target', 'Weight'])
    # Export Edges csv
    edge_df.to_csv(edges_path, index=False)
    return


def check_for_at_least_two_text_files(dirname):
    '''
    Checks whether there are two text files in the directory
    Parameters
    ----------
    dirname : the name of the directory to check

    Returns whether or not the directory contains at least two text files
    '''
    # Get a list of all files in the chosen directory
    my_list = os.listdir(dirname)
    # Create counter
    counter = 0
    # Iterate over the list to get all filenames that end with .txt
    for filename in my_list:
        if filename.endswith('.txt'):
            counter +=1
    if counter >= 2:
        return True
    else:
        return False 
    
                

##############################################################################

# Variables necessary to run program
directory_exists = False
texts_exist = False
script_location = os.path.dirname(os.path.abspath(__file__))

# Perform the loop until we find a directory with texts in it
while directory_exists == False or texts_exist == False:
    print("Please enter a filepath (relative to current directory):")
    text_input = input()
    # Get current directory
    cwd = os.getcwd() + "\\"+ text_input + "\\"
    # Check if the directory exists
    directory_exists = os.path.isdir(cwd)
    if directory_exists == True:
        texts_exist = check_for_at_least_two_text_files(cwd)
    

# Notify user of what's happening
print('Creating Frequency Distributions')
# Create frequency distributions, return their location
freq_dist_location = freqdist(cwd)
# Do consensus
out_path = create_consensus_matrix(freq_dist_location)
# Write edges and nodes csvs
create_edge_and_node_files(out_path)
print('Done')
