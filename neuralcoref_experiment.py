# Load your usual SpaCy model (one of SpaCy English models)
import spacy
nlp = spacy.load('en_core_web_sm')
import logging
logging.basicConfig(level=logging.INFO)

# Add neural coref to SpaCy's pipe
import neuralcoref
# neuralcoref.add_to_pipe(nlp)
coref = neuralcoref.NeuralCoref(nlp.vocab)
# nlp.add_pipe(coref, name="neuralcoref")

# You're done. You can now use NeuralCoref as you usually manipulate a SpaCy document annotations.
doc = nlp(u'My sister has a dog. She loves him.')

print(doc._.has_coref)
print(doc._.coref_clusters)