try:
    import unzip_requirements
except ImportError:
    pass
import json
import nltk
import pandas as pd
nltk.data.path.append("/tmp")
nltk.download("punkt", download_dir = "/tmp")
nltk.download('averaged_perceptron_tagger', download_dir = "/tmp")
nltk.download('maxent_ne_chunker', download_dir = "/tmp")
nltk.download('words', download_dir = "/tmp")

hotel_names = pd.read_csv('data/hotel_names.txt', sep='\t')

def get_entities(event, context):
    try:
        body = json.loads(event['body'])
        hotels = get_hotels(body['sentence'])
        #TODO
        features = ""
        aspect = ""

        return {
        "statusCode": 200,
        "headers": {
        'Content-Type': 'application/json',
        'Access-Control-Allow-Origin': '*',
        "Access-Control-Allow-Credentials": True

        },
        "body": json.dumps({'hotels':hotels, 'features':features, 'aspect':aspect})
        }
    except Exception as e:
        return {
        "statusCode": 500,
        "headers": {
        'Content-Type': 'application/json',
        'Access-Control-Allow-Origin': '*',
        "Access-Control-Allow-Credentials": True
        },
        "body": json.dumps({"error": repr(e)})
        }

def get_hotels(sentence):
    chunks = []
    answerToken = nltk.word_tokenize(sentence)
    nc = nltk.ne_chunk(nltk.pos_tag(answerToken))
    entity = {"label": None, "chunk": []}
    for c_node in nc:
        if (type(c_node) == nltk.Tree):
            if (entity["label"] == None):
                entity["label"] = c_node.label()
                entity["chunk"].extend([token for (token, pos) in c_node.leaves()])
        else:
            (token, pos) = c_node
            if pos == "NN" or pos == "JJ":
                # search in list of hotels, to discard it relates to other NNs, e.g. restaurant. Case when names of hotels are in lower case
                # JJ is special case when refer to hotel without naming the word hotel, e.g. "the Julian"
                complete = hotel_names[hotel_names['name'].str.contains(token.capitalize())]['complete']
                if len(complete) > 0:
                    entity["chunk"].append(complete[0])
            if pos == "NNP":
                entity["chunk"].append(token)
            else:
                if not len(entity["chunk"]) == 0:
                    # chunks.append((entity["label"]," ".join(entity["chunk"])))
                    chunks.append((" ".join(entity["chunk"])))
                    entity = {"label": None, "chunk": []}
    if not len(entity["chunk"]) == 0:
        chunks.append((entity["label"], " ".join(entity["chunk"])))
    return chunks
