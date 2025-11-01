from nltk.corpus import wordnet as wn
from nltk.corpus import words





def how_much_we_have_words():
    return len(words.words())


def find_synsets(word : str ):
    return wn.synsets(word)


def main():
    res = find_synsets("power")

    for w in res : 
        print(f"meanong of power is  : {w} -> {w.definition()}")
        print(f"example ::::: {w.examples()}")






main()