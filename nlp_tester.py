# import spacy
# nlp = spacy.load("en_core_web_sm")
#
#
# cataphora_documents_list = ["Because he was very cold, David put on his "
#                               "coat.",
#                       "His friends have been criticizing Jim for "
#                       "exaggerating.",
#                       "Although Sam might do so, I shall not buy a new bike.",
#                       "In their free time, the boys play video games."]
#
#
# anaphora_documents_list = [["Susan dropped the plate, it shattered loudly.",
#                      "The music stopped, and that upset everyone.",
#                      "Fred was angry, and so was I.",
#                      "If Sam buys a new bike, I will do it as well."][0]]
# anaphora_documents_list = ["Tom and Jerry are running around, but none of them can catche the other"]
#
# for sentence in anaphora_documents_list:
#     doc = nlp(sentence)
#     for np in doc.noun_chunks:
#         print(np.text)
#     for token in doc:
#         print(token, token.pos_, token.dep_)
#
#
# tom = " ".join(['PROPN', 'nsubj', 'PRON', 'pobj'])
# jerry = " ".join(['PROPN', 'conj', 'PRON', 'pobj'])
#
# data =[tom, jerry]
# from joblib import load
# from sklearn.ensemble import AdaBoostClassifier
# from sklearn.feature_extraction.text import CountVectorizer
# model = load("best_anaphora_model.joblib")
# cv = load("anaphora_vectorizer.joblib")
#
# vec = cv.transform(data)
# print(vec.shape)
# print(model.predict_proba(vec))
#
# """
# EURISTIC FILTERING CRITERIAS
#
# anaphora: it -> the antedent can't be a PROPN
#              -> use the hiponyms of the words human, man, woman, child,
#
# select noun chunks as candidates for a anaphora model
# """
#

import numpy as np
from nltk.corpus import wordnet as wn
object_synset = wn.synset('object.n.01')
object_hyponyms_set = set(list(set([w for s in object_synset.closure(lambda
                                                                      s:s.hyponyms()) for w
                           in s.lemma_names()])))

print(len(object_hyponyms_set))
print("bike" in typesOfVehicles)

np.save("object_hyponyms.npy", np.array(typesOfVehicles), allow_pickle=True)
object_hyponyms_set = set(list(np.load("object_hyponyms.npy",
                                       allow_pickle=True)))

print(len(typesOfVehicles))