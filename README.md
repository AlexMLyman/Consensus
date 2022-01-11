# Consensus

Python script for creating consensus networks based on Maciej Eder's paper, **Visualization in Stylometry: Cluster Analysis Using Networks**.
Paper can be found [here](https://academic.oup.com/dsh/article/32/1/50/2957386).

## Getting Started

This should be rather easy to use, assuming you have python installed on your computer. The steps are as follows:
1. Gather your files
2. Install dependencies
3. Run the network creation script

### Gathering your Files

The script performs all necessary pre-processing tasks, so all you need to provide is a folder of .txt files. (preferably utf-8, but most encodings should work)
You can use books, chapters, essays, or really any texts you wish. It helps if your files have descriptive names.

### Installing Dependencies

After downloading the requirements.txt file, just run something like `pip install -r requirements.txt` from the command line. 
You can also install the dependencies individually using pip if you wish.

### Running the Script

Running the script is extremely simple. Just run the script *something like* `python consensus.py` *should do the trick.*.
You will be prompted for a filepath **Relative to the script's location**. After that, hit enter and the network will generate.
You'll be given two .csv files, one containing the nodes (each representing one of your text files) and the edges between them.
These are ready to import into force-directed graphing software.

## Advanced

You can tune parameters with how many layers in the `create_different_length_docs` function. 
You can also change how many nearest neighbors to select by changing the value of n in the  `perform_consensus` function definition.

### Examples

Check out an example network, created using the 85 Federalist Papers [here](https://alexlyman.org/federalist/).
