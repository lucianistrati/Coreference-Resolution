from sklearn.ensemble import AdaBoostClassifier
from sklearn.feature_extraction.text import CountVectorizer
from nltk.corpus import wordnet as wn

from copy import deepcopy
from joblib import load, dump
import warnings
warnings.filterwarnings("ignore")
import numpy as np

import spacy

nlp = spacy.load("en_core_web_sm")

object_synset = wn.synset('object.n.01')
# vom considera doar primul sens pentru substantivul "object"

object_hyponyms_set = set(list(set([w for s in
                                    object_synset.closure(lambda s:s.hyponyms())
                                    for w in s.lemma_names()])))
# substantivul "object" are 47047 de hiponime, probabil majoritatea cuvintelor
# care pot fi coreferentiate prin pronumele obiectual "it" se afla in aceasta
# multime de cuvinte


vectorizer = load("anaphora_vectorizer.joblib")
model = load("best_anaphora_model.joblib")


def is_not_coreference(first_token, second_token, first_token_pos,
                                second_token_pos):
    """
    Nu are sens sa coreferentiem un substantiv propriu cu persoana intai
    singular sau cu pronumele obiectual 'it'
    """
    if first_token_pos == "PRON" and second_token_pos == "PROPN":
        if first_token.lower() == "it" or first_token.lower() == "i":
            return True

    """
    Daca referentiem un substantiv comun (ce se refera la un obiect) printr-un 
    pronume, atunci cu siguranta acel pronume ar trebui sa fie pronumele 
    obiectual 'it'
    """
    if first_token_pos == "PRON" and second_token_pos == "NOUN":
        if first_token.lower() != 'it' and second_token.lower() in \
                object_hyponyms_set:
            return True

    return False


def extract_features_from_text(text):
    """
    extragem trasaturile pentru toate combinatiile de tip PRONUME-SUBSTANTIV,
    intrucat in setul de date de antrenare exista doar combinatii de acest tip
    """
    doc = nlp(text)

    token_tags_list = []
    token_candidates_list = []

    for token in doc:
        if token.pos_ == "PRON":
            token_tags = ""
            token_tags += (token.dep_ + " " + token.pos_ + " " + "_")
            for token2 in doc:
                if token2.pos_ == "PROPN" or token2.pos_ == "NOUN":
                    if is_not_coreference(token.text, token2.text, token.pos_, \
                                          token2.pos_) == True:
                        continue

                    token_tags_compound = token_tags + (token2.pos_ + " " +
                                                token2.dep_)
                    token_tags_list.append(token_tags_compound)
                    token_candidates_list.append((token.text, token2.text))

    return token_tags_list, token_candidates_list


def multiple_documents_infer():
    """
    inferam cu ajutorul modelului de coreferinta pe 4 propozitii de tip
    anafora - antecedent si catafora - postcedent
    """
    cataphora_documents_list = ["Because he was very cold, David put "
                                "on his "
                              "coat.",
                      "His friends have been criticizing Jim for "
                      "exaggerating.",
                      "Although Sam might do so, I shall not buy a new bike.",
                      "In their free time, the boys play video games."]

    cataphora_list = ["he", "his", "do so", "their"]
    postcedent_list = ["David", "Jim", "buy a new bike", "the boys"]
    c_reasons_list = ["RIGHT_PRED", "NO_DET_IN_TRAINING_SET",
                      "NO_VERBS_IN_TRAINING_SET",
                      "NO_DET_IN_TRAINING_SET"]

    for i, cataphora_doc in enumerate(cataphora_documents_list):
        print("*" * 5)
        print("actual (postcedent - cataphora): (" + postcedent_list[i] + ", "
              + cataphora_list[i] + ")")
        token_tags_list, token_candidates_list = extract_features_from_text(
            cataphora_doc)
        if len(token_candidates_list) and len(token_tags_list):
            for token_tags in token_tags_list:
                vectorized_data = vectorizer.transform(token_tags.split())
            probas_list = model.predict_proba(vectorized_data)
            valid_probas_list = probas_list[:, 1]

            coref_idx = np.argmax(valid_probas_list)

            if coref_idx >= len(valid_probas_list)  or \
                    coref_idx >= len(token_candidates_list):
                coref_idx = 0

            if valid_probas_list[coref_idx] > .5:
                print("Predicted coreference: ", token_candidates_list[
                    coref_idx])
            else:
                print("No coreference was found!")
        else:
            print("No coreference was found!")
        print(c_reasons_list[i])
        print("*" * 5)

    anaphora_documents_list = ["Susan dropped the plate, it shattered loudly.",
                     "The music stopped, and that upset everyone.",
                     "Fred was angry, and so was I.",
                     "If Sam buys a new bike, I will do it as well."]

    antecedent_list = ["the plate", "The music stopped", "angry", "buys a new bike"]
    anaphora_list = ["it", "that", "so", "do it"]
    a_reasons_list = ["RIGHT_PRED", "HALF_RIGHT_PRED",
                      "NO_ADJECTIVES_IN_TRAINING_SET",
                      "RIGHT_PRED"]

    for i, anaphora_doc in enumerate(anaphora_documents_list):
        print("*" * 5)
        print("actual (antecedent - anaphora): (" + antecedent_list[i] + ", "
              + anaphora_list[i] + ")")
        token_tags_list, token_candidates_list = extract_features_from_text(
            anaphora_doc)

        if len(token_candidates_list) and len(token_tags_list):
            for token_tags in token_tags_list:
                vectorized_data = vectorizer.transform(token_tags.split())
            probas_list = model.predict_proba(vectorized_data)
            valid_probas_list = probas_list[:, 1]
            coref_idx = np.argmax(valid_probas_list)
            if coref_idx >= len(valid_probas_list) or \
                    coref_idx >= len(token_candidates_list):
                coref_idx = 0

            if valid_probas_list[coref_idx] > .5:
                print("Predicted coreference: ", token_candidates_list[
                    coref_idx])
            else:
                print("No coreference was found!")
        else:
            print("No coreference was found!")
        print(a_reasons_list[i])
        print("*" * 5)


def single_document_infer(document):
    """
    inferam cu ajutorul modelului de identificare a coreferintelor pe un
    singur document
    """
    print("*" * 5)
    token_tags_list, token_candidates_list = extract_features_from_text(document)
    pronoun, coref_noun = None, None
    if len(token_candidates_list) and len(token_tags_list):
        for token_tags in token_tags_list:
            vectorized_data = vectorizer.transform(token_tags.split())
        probas_list = model.predict_proba(vectorized_data)
        valid_probas_list = probas_list[:, 1]

        coref_idx = np.argmax(valid_probas_list)

        if coref_idx >= len(valid_probas_list)  or \
                coref_idx >= len(token_candidates_list):
            coref_idx = 0

        if valid_probas_list[coref_idx] > .5:
            pronoun = token_candidates_list[coref_idx][0]
            coref_noun = token_candidates_list[coref_idx][1]
            print("Predicted coreference: ", token_candidates_list[
                coref_idx])
        else:
            print("No coreference was found!")
    else:
        print("No coreference was found!")

    if pronoun and coref_noun:
        print("Original text:")
        print("'" + document + "'")
        print("Replaced pronoun with coreferenced noun:")
        print("'" + document.replace(pronoun, coref_noun) + "'")

    print("*" * 5)


def main():
    # multiple_documents_infer()
    document = "Susan dropped the plate, it shattered loudly."
    single_document_infer(document)

if __name__=='__main__':
    main()