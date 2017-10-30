import arff
import scipy
import os
import json
import pprint
# import read_tsv
pp = pprint.PrettyPrinter(indent=4)

basepath = "/home/bt1/13CS10060/btp"
dataset_path = os.path.join(basepath, "codes","dataset")


def verify(set1, set2):
    n = len(set1)
    for i in range(n):
        if(i == 1 or i == 3 or i == 0):
            continue
        if(set1[i].strip("\"\'") != set2[i].strip("\"\'")):
            if(set1[i] == '?' and set2[i] == ''):
                continue
            # print(set1[i].strip("\"\'"), set2[i].strip("\"\'"))
            return False
    return True


def merge_two(dataset1, dataset2, intro_offset):
     
    merged_attris = dataset1['attributes'][:-1]
    merged_attris.extend(dataset2['attributes'][intro_offset:]) 


    n = len(dataset1['data'])
    m = len(dataset2['data'])
    print(n,m)
    dataset = []

    for i in range(min(n,m)):
        header1 = dataset1['data'][i][:intro_offset]
        header2 = dataset2['data'][i][:intro_offset]

        if(not verify(header1, header2)):
            print(dataset1['data'][i], dataset2['data'][i])
            return
            

        instance = dataset1['data'][i][:-1]
        instance.extend(dataset2['data'][i][intro_offset:])

        dataset.append(instance)

    return {"attributes": merged_attris, "data" :dataset}


def merge_all(datasets, offset):

    merged = datasets[0]
    for dset in datasets[1:]:
        merged = merge_two(merged, dset, offset)

    return merged



def remove_extra(dataset):
    names = []

    for n in dataset['header']['attributes']:
        names.append(n['name'])


    dset = []
    for ins in dataset['data']:
        dset.append(ins['values'])
    return {"attributes": names, "data": dset}


def merge_past():

    all_data =  json.load(open(dataset_path + "/all470.json"))
    dpast = json.load(open(dataset_path + "/dataset_past.json"))

    attributes = all_data['attributes'][:-1]
    attributes.append("past")
    attributes.append(all_data['attributes'][-1])

    dataset = []
    for i in range(len(dpast['data'])):
        past_value = dpast['data'][i][-1]
        if(past_value == "True" or past_value == "true" or past_value == True):
        	past_value = 1
        if(past_value == "False" or past_value == "false" or past_value == False):
        	past_value = 0
        header_past = dpast['data'][i][-4:-1]
        header_all = all_data['data'][i][6:9]
        ad = all_data['data'][i][:-1]
        ad.append(past_value)
        ad.append(all_data['data'][i][-1])
        for j in range(3):
            if(header_all[j] != header_past[j]):
                print(all_data['data'][i])
                break
        dataset.append(ad)

    merged = {"attributes": attributes, "data": dataset}
    json.dump(merged, open(os.path.join(dataset_path, "all471.json"), "w"))
    pp.pprint(merged['attributes'])
    print(len(merged['data']), len(merged['data'][0]))


def merge_buster_score():
    all_data =  json.load(open(dataset_path + "/all471_with_buster.json"))
    tfidf = json.load(open("tfidf.json"))
    OFFSET = 9
    attributes = all_data['attributes'][:-1]
    for a in tfidf['attributes'][OFFSET:-1]:
        attributes.append(a[0])
    # attributes.extend(tfidf['attributes'][OFFSET:-1])
    attributes.append(all_data['attributes'][-1])

    dataset = []
    for i in range(len(all_data['data'])):
        alli = all_data['data'][i]
        tfidfi = tfidf['data'][i]
        
        header_tfidf = tfidfi[6:9]
        header_all = alli[6:9]
        ad = all_data['data'][i][:-1]
        for j in range(OFFSET, len(ad)-1):
            ad[j] = float(ad[j])
        ad.extend(tfidfi[OFFSET:-1])
        cl = tfidfi[-1]
        if(cl != int(all_data['data'][i][-1])):
            print(all_data['data'][i], cl)
            break
        ad.append(all_data['data'][i][-1])
        for j in range(3):
            if(int(header_all[j]) != int(header_tfidf[j])):
                print(header_all, header_tfidf, j)
                break
        dataset.append(ad)

    merged = {"attributes": attributes, "data": dataset}
    json.dump(merged, open(os.path.join(dataset_path, "all471+tfidf+buster.json"), "w"))
    # pp.pprint(merged['attributes'])
    print(len(merged['data']), len(merged['data'][0]))



def main():
    dnames = ["entity", "liwc", "dataset_topic", "similarity_tfidf_1gram", "data_all"]
    # dnames = ["dataset_past", "all470"]
    datasets = []
    for n in dnames:
        dset = json.load(open(dataset_path + "/" + n +".json"))
        datasets.append(dset)#)
        print (n)
    # datasets = [for n in dnames]

    datasets = [remove_extra(d) for d in datasets]
    # datasets_data = [d['data'] for d in datasets]
    offset = 9
    merged = merge_all(datasets, offset)
    json.dump(merged, open(os.path.join(dataset_path, "all470.json"), "w"))
    pp.pprint(merged['attributes'])
    print(len(merged['data']), len(merged['data'][0]))


merge_buster_score()
# main()
