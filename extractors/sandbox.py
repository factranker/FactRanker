from extractors import tokenizer
import pprint
import json
import pickle as pkl
pp = pprint.PrettyPrinter(indent=4)

def past_tense(sentence):
    global pp
    sentence = sentence.strip()
    text = ( sentence )
    output = tokenizer.dependency_parse(text)['sentences'][0]
    root = output['basic-dependencies'][0]
    assert root['dep'] == 'ROOT'
    index = root['dependent']
    token = output['tokens'][index-1]
    assert token['index'] == index
    pos = token['pos']
    # print(pos)
    if pos=='VBD' or pos=='VBN':
        return True
    return False
    # pp.pprint(output['sentences'][0]['parse'])

def main():
    correct = 0
    incorrect = 0
    try:
        raise Exception
        data_feat = pkl.load(open('past_pickle.pkl','rb'))

    except:
        with open('/home/bt1/13CS10060/btp/ayush_dataset/annotated_single_all.tsv','r') as annotated_file:
            annotated_file = annotated_file.readlines()
        i = 1
        total = 0
        rows = []
        for row in annotated_file[1:]:
            try:
                row = [x.strip() for x in row.split('\t')][:9]
            except:
                print('row', total, 'Had < 9 columns')
            rows.append((row[0],row[1]))
        window_size = 2
        l = len(annotated_file[1:])
        data_feat = []
        while i<=l:
            try:
                row = [x.strip() for x in annotated_file[i].split('\t')][:9]
            except:
                print('row', total, 'Had < 9 columns')
            past = True
            for j in range(max(0,i-window_size),min(i-window_size+1,l)):
                if not past_tense(rows[j][1]):
                    past = False
            row.append(past)
            if past:
                if row[0].lower()[0]=='y':
                    correct+=1
                else:
                    incorrect+=1
                print(correct*1.0/(correct+incorrect))
            data_feat.append(row)
            print(i)
            i+=1


        pkl.dump(data_feat, open('past_pickle.pkl','wb'))

    attributes = ['Checked', 'Sentence', 'Marked', 'By','Speaker', 'Party', 'DebateId', 'ID', 'Id_1', 'Past']
    dataset = {
            'description': 'data_file',
            'relation': 'statements',
            'attributes': attributes,
            'data': data_feat
            }

    json_data = {'attributes' : attributes, 'data' : data_feat}
    json.dump(json_data,open('dataset_past.json','w'))
    filename_data = "dataset_past"+".arff"  
    f = open(filename_data,'w')
    f.write(arff.dumps(dataset))
    f.close()
