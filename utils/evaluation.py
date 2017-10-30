def evaluate(X_test, y_test, clf, name, sent_print=True):

    y_hat = clf.predict(X_test)
    report = metrics.classification_report(y_test, y_hat)
    
    print(report)

    try:
        y_prob = clf.predict_proba(X_test)[:,1]
    except:
        pass

    ks = [10,20,30,40,50,60,70,80,90,100,200,300,500,1000]

    allscores = rank_scorers.all_score(y_test, y_prob, ks)

    
        
    for i,k in enumerate(ks):
        print(k,round(allscores[i][0],3),round(allscores[i][1],3),round(allscores[i][2],3), sep="\t", file=f)