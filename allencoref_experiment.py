from allennlp.predictors.predictor import Predictor
import allennlp
# allennlp.common.plugins.load_plugins()

# predictor = Predictor.from_path("https://s3-us-west-2.amazonaws.com/allennlp/models/coref-model-2018.02.05.tar.gz")
predictor = Predictor.from_path("models/model.tar.gz",
                                  predictor_name="textual_entailment")

sent = "my mother was happy. She gave me a pat on the back" #or any other sentence.
pred = predictor.predict(document=sent)
clusters = pred['clusters']
document = pred['document']

n = 0
doc = {}
for obj in document:
    doc.update({n :  obj}) #what I'm doing here is creating a dictionary of each word with its respective index, making it easier later.
    n = n+1

clus_all = []
cluster = []
clus_one = {}
for i in range(0, len(clusters)):
    one_cl = clusters[i]
    for count in range(0, len(one_cl)):
        obj = one_cl[count]
        for num in range((obj[0]), (obj[1]+1)):
            for n in doc:
                if num == n:
                    cluster.append(doc[n])
    clus_all.append(cluster)
    cluster = []
print (clus_all) #And finally, this shows all coreferences