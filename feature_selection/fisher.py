import scipy.io
import numpy as np
import os

from sklearn import linear_model
from sklearn.model_selection import train_test_split
from skfeature.function.similarity_based import fisher_score
from skfeature.function.information_theoretical_based import CIFE
from sklearn.ensemble import ExtraTreesClassifier,AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_selection import SelectFromModel,mutual_info_classif,chi2
from sklearn.preprocessing import MinMaxScaler
from sklearn import svm
import util
from sklearn.metrics import recall_score,confusion_matrix,f1_score,accuracy_score
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from itertools import combinations
from collections import defaultdict,Counter
import matplotlib.pyplot as plt 
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
import random 
import calculate_mutual_info
import mifs

# mat = scipy.io.loadmat("COIL20.mat")

# X=mat['X']

# y = mat['Y'][:, 0]
# n_samples, n_features = np.shape(X)


# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=40)
# score = fisher_score.fisher_score(X_train, y_train)

# idx = fisher_score.feature_ranking(score)
# num_fea = 5
# selected_features_train = X_train[:, idx[0:num_fea]]
# selected_features_test = X_test[:, idx[0:num_fea]]
# clf = svm.LinearSVC()

# clf.fit(selected_features_train, y_train)

# y_predict = clf.predict(selected_features_test)
# acc = accuracy_score(y_test, y_predict)
# print(acc)

def read_feature_name(feature_file):
    f=open(feature_file,"r")
    content=f.readlines()
    f.close()
    feature_names=[]
    for row in content:
        feature_names.append(row.split()[1])
    return feature_names

def select_data(class1,class2,embo,labels,one_vs_one=True):
    curr_index1=np.argwhere(labels==class1)
    if one_vs_one:
        curr_index2=np.argwhere(labels==class2)
    else:
        assert class2==-1
        curr_index2=np.argwhere(labels!=class1)
    curr_embo=np.concatenate((embo[curr_index1],embo[curr_index2])).squeeze()
    curr_labels=np.concatenate((labels[curr_index1],labels[curr_index2])).squeeze()
    curr_labels[:len(curr_index1)]=0
    curr_labels[len(curr_index1):]=1
    return curr_embo,curr_labels

def compute_fisher_score(train_features,train_labels):
    labels=Counter(train_labels)
    num_features = train_features.shape[1]
    num_classes = len(labels.keys())
    indexes = []
    for i in range(num_classes):
        indexes.append(np.argwhere(train_labels==i))
    features_score = np.zeros(num_features)
    for i in range(num_features):
        mu_i=np.mean(train_features[:,i])
        num=0
        den=1e-6
        for j in range(num_classes):
            mu_ij=np.mean(train_features[indexes[j],i])
            var_ij=np.var(train_features[indexes[j],i])
            num+=labels[j]*np.square(mu_ij-mu_i)
            den+=labels[j]*var_ij
        features_score[i]=num/den
    idx = np.argsort(features_score)[::-1]
    return features_score,idx

def compute_extratree_score(train_features,train_labels):
    clf = ExtraTreesClassifier(random_state=0)
    clf=clf.fit(train_features,train_labels)
    pred = clf.predict(test_features)
    print("Extratree classifier testing dataset")
    acc = accuracy_score(test_labels,pred)
    mf1 = f1_score(test_labels,pred,average="macro")
    wf1 = f1_score(test_labels,pred,average="weighted")
    conf = confusion_matrix(test_labels,pred)
    print("acc",acc)
    print("macro",mf1)
    print("weighted",wf1)
    print("Confusion matrix")
    print(conf)
    score = clf.feature_importances_
    idx = np.argsort(score)[::-1]
    return score,idx

def compute_cife_score(train_features,train_labels):
    # no score available for CIFE extraction
    idx,score,_ = CIFE.cife(train_features,train_labels,n_selected_features=10)
    return score,idx



def compute_mi_score(train_features,train_labels):
    
    mi = mutual_info_classif(train_features,train_labels,random_state=0)
    idx = np.argsort(mi)[::-1]
    # mi = []
    # _,n_features = train_features.shape
    # for i in range(n_features):
    #     print("current feature",i)
    #     mi.append(calculate_mutual_info.midd(train_features[:,i],train_labels))
    # idx = np.argsort(mi)[::-1]
    return mi,idx

def compute_mrmr_score(train_features,train_labels):
    train_labels=np.asarray(train_labels,dtype=np.int64)
    feat_selector = mifs.MutualInformationFeatureSelector(n_features=100, \
        method='MRMR', verbose=2, n_jobs=-1)
    feat_selector.fit(train_features,train_labels)
    score = np.asarray(feat_selector.mi_)
    idx = np.asarray(feat_selector.ranking_)
    #score = score[idx]
    return score,idx

def save_mrmr_score(fold):
    f=open("mrmr_{}.txt".format(fold),"r")
    content=f.readlines()
    f.close()
    idx=[]
    score=[]
    for i in range(len(content)):
        idx.append(int(content[i].split(":")[1].split(",")[0]))
        score.append(float(content[i].split(":")[2].strip("\n")))
    return score,idx

def compute_l1(train_features,train_labels):
    #clf = linear_model.Lasso(alpha=0.1)
    clf = svm.LinearSVC(C=0.01,penalty="l1",dual=False).fit(train_features,train_labels)
    selector = SelectFromModel(clf).fit(train_features,train_labels)
    score = selector.estimator_.coef_
    idx = np.argsort(score)[::-1]
    return score, idx

def compute_chi2(train_features,train_labels):
    scalar = MinMaxScaler()
    scalar.fit(train_features)
    train_features= scalar.fit_transform(train_features)
    score,_ = chi2(train_features,train_labels)
    idx = np.argsort(score)[::-1]
    return score, idx

def compute_features(train_features,train_labels,path,feature_num=1000,method="fisher"):
    if method=="fisher":
        score, idx = compute_fisher_score(train_features,train_labels)
    elif method == "extratree":
        score, idx = compute_extratree_score(train_features,train_labels)
    elif method == "mi":
        score, idx = compute_mi_score(train_features,train_labels)
    elif method == "l1":
        score, idx = compute_l1(train_features,train_labels)
    elif method == "mrmr":
        score,idx = compute_mrmr_score(train_features,train_labels)
        #score,idx = save_mrmr_score(fold)
    elif method == "chi2":
        score, idx = compute_chi2(train_features,train_labels)

    importance = np.c_[(score,idx)].T
    if path!="":
        print(target_folder,path)
        np.save(os.path.join(target_folder,path),importance)
    return score,idx

def load_features(path):
    fisher_scores = np.load(os.path.join(target_folder,path))
    try:
        score = np.asarray(fisher_scores[0,:],dtype=np.float)
        idx = np.asarray(fisher_scores[1,:],dtype=np.int16)
    except:
        idx = np.asarray(fisher_scores,dtype=np.int16)
        score = np.zeros_like(idx)
    return score,idx

def perform_LDA(train_features_selected,train_labels,test_features_selected,test_labels,weighted=True):
    priors=None
    if weighted: 
        counter=Counter(train_labels)
        unique_classes=np.unique(train_labels)
        priors=np.zeros_like(unique_classes)
        for c in unique_classes:
            priors[c]=len(train_labels)/counter[c]
    clf = LDA(n_components=None, priors=priors, shrinkage=None,
        solver='svd', store_covariance=False, tol=0.0001)
    clf = clf.fit(train_features_selected,train_labels)
    pred=clf.predict(test_features_selected)
    acc = accuracy_score(test_labels,pred)
    mf1 = f1_score(test_labels,pred,average="macro")
    wf1 = f1_score(test_labels,pred,average="weighted")
    uar = recall_score(test_labels,pred,average="macro")
    conf = confusion_matrix(test_labels,pred)
    print("acc",acc,"weighted f1",wf1,"macro F1",mf1,"uar",uar)
    print("Confusion matrix")
    print(conf)
    return acc, mf1, wf1, conf

def load_feature_sets(fisher_file,feature_num=10,save=True,padded=False):
    print("feature num",feature_num)
    fisher_scores=np.load(os.path.join(target_folder,"{}_{}.npy").format(fisher_file,fold))
    #fisher_scores=np.load(os.path.join(target_folder,"{}_{}.npy").format(fisher_file,feature_name))
    idx = np.asarray(fisher_scores[1,:],dtype=np.int16)

    if padded:
        train_features_selected = np.zeros((len(train_features),1600))
        test_features_selected = np.zeros((len(test_features),1600))

        for i in range(feature_num):
            train_features_selected[:,i] = train_features[:,idx[i]]
            test_features_selected[:,i] = test_features[:,idx[i]]   
    else:
        train_features_selected = train_features[:,idx[:feature_num]]
        test_features_selected = test_features[:,idx[:feature_num]]

    print(train_features_selected.shape)
    if save:
        if padded:
            util.write_h5(train_features_selected,os.path.join(target_folder,"train_{}_{}_{}_padded.h5".format(feature_name,fisher_file,feature_num)))
            util.write_h5(test_features_selected,os.path.join(target_folder,"test_{}_{}_{}_padded.h5".format(feature_name,fisher_file,feature_num)))
        else:
            util.write_h5(train_features_selected,os.path.join(target_folder,"train_{}_{}_{}.h5".format(feature_name,fisher_file,feature_num)))
            util.write_h5(test_features_selected,os.path.join(target_folder,"test_{}_{}_{}.h5".format(feature_name,fisher_file,feature_num)))

    return perform_LDA(train_features_selected, train_labels, test_features_selected,test_labels)

def train_model_one_versus_one(class1,class2,feature_num=100,method="extratree"):
    print("target label",combo[0][0],combo[1][0])
    curr_training_features,curr_training_labels=select_data(class1,class2,train_features,train_labels)
    curr_testing_features,curr_testing_labels=select_data(class1,class2,test_features,test_labels)
    score,idx = compute_features(curr_training_features,curr_training_labels,"{}_{}_{}_{}.npy".format(method,combo[0][0],combo[1][0],fold),feature_num,method)
    #score,idx = load_features("{}_{}_{}_{}.npy".format(method,combo[0][0],combo[1][0],fold))
    
    train_features_selected = curr_training_features[:,idx[:feature_num]]
    test_features_selected = curr_testing_features[:,idx[:feature_num]]

    return perform_LDA(train_features_selected,curr_training_labels,test_features_selected,curr_testing_labels)

def train_model_one_versus_all(class1,feature_num=400,method="extratree"):
    print("target label",idx2labels[class1])
    curr_training_features,curr_training_labels=select_data(class1,-1,train_features,train_labels,False)
    curr_testing_features,curr_testing_labels=select_data(class1,-1,test_features,test_labels,False)
    score,idx = compute_features(curr_training_features,curr_training_labels,"{}_{}_{}.npy".format(method,idx2labels[class1],fold),feature_num,method)
    #score,idx = load_features("{}_{}_{}.npy".format(method,idx2labels[class1],fold))

    train_features_selected = curr_training_features[:,idx[:feature_num]]
    test_features_selected = curr_testing_features[:,idx[:feature_num]]

    return perform_LDA(train_features_selected,curr_training_labels,test_features_selected,curr_testing_labels)

def output_feature_names(fisher_file,feature_num=30,method="mrmr"):
    content=[]
    dict_scores=defaultdict(float)

    for j in range(1,4):
        score,idx = load_features("{}_{}.npy".format(fisher_file,j))
        Z= np.sum(np.exp(score))
        score = np.exp(score)/Z
        content.append("fold {}\n".format(j))
        if method!="mrmr":
            for i in range(feature_num):
                dict_scores[idx[i]]+=score[idx[i]]
                content.append("{}\t{}\t{}\n".format(idx[i],feature_names[idx[i]],score[idx[i]]))
        else:
            for i in range(feature_num):
                dict_scores[idx[i]]+=score[i]
                content.append("{}\t{}\t{}\n".format(idx[i],feature_names[idx[i]],score[i]))

    content.append("\n")
    sorted_dict=dict(sorted(dict_scores.items(), key=lambda item: item[1],reverse=True))
    for key,val in sorted_dict.items():
        content.append("{}\t{}\t{}\n".format(key,feature_names[key],val))
    return content

def get_most_important_features(file,idx=False,feature_num=30):
    f=open(file,"r")
    content=f.readlines()
    f.close()
    split_idx=content.index("\n")
    content=content[split_idx+1:split_idx+feature_num+1]
    #content=content[split_idx+1:]
    if not idx:
        return_content = [row.split("\t")[1] for row in content]
    else:
        return_content = [int(row.split("\t")[0]) for row in content]
    return return_content

def find_over_lap_features(method="fisher",target_class="CRY",idx=False):
    #print('target class',target_class)
    features_dict={}
    target_features = get_most_important_features(os.path.join(method,"{}_feature_importance_{}.txt".format(method,target_class)),idx)
    for key in labels_dict.keys():
        if key!=target_class:
            try:
                features = get_most_important_features(os.path.join(method,"{}_feature_importance_{}_{}.txt".format(method,key,target_class)),idx)
            except:
                features = get_most_important_features(os.path.join(method,"{}_feature_importance_{}_{}.txt".format(method,target_class,key)),idx)
            features_dict[key] = [x for x in target_features if x in frozenset(features)]
    # for key,val in features_dict.items():
    #     print(key)
    #     for v in val[:30]:
    #         print(v)
    # print("\n")
    return features_dict

def get_num_samples(test_labels,bab_num=600,other_num=120):
    dict_idx_samples={}
    for i in range(len(test_labels)):
        if test_labels[i] not in dict_idx_samples:
            dict_idx_samples[test_labels[i]]=[]
        dict_idx_samples[test_labels[i]].append(i)
    out_idx=[]
    for idx in idx2labels.keys():
        #random.shuffle(dict_idx_samples[idx])
        # if idx == 1:
        #     out_idx+=dict_idx_samples[idx][:fus_num]
        if idx == 3:
            out_idx+=dict_idx_samples[idx][:bab_num]
        else:
            out_idx+=dict_idx_samples[idx][:other_num]
    return np.asarray(out_idx)

def PCA_analysis(feature_num=5,fold=1,method="mrmr",cluster_method="one-vs-one"):
    all_idx=[]
    feature_name="combined{}_multiple_norm".format(fold)
    if cluster_method=="one-vs-one":
        for count,combo in enumerate(combinations(labels_dict.items(),2)):
            score,idx = load_features("{}_{}_{}_{}.npy".format(method,combo[0][0],combo[1][0],fold))
            all_idx.extend(idx[:feature_num])
            #all_idx.extend(get_most_important_features(os.path.join(method,"{}_feature_importance_{}_{}.txt".format(method,combo[0][0],combo[1][0])),idx=True)[:feature_num])
    elif cluster_method=="one-vs-all":
        for key in labels_dict.keys():
            score,idx = load_features("{}_{}_{}.npy".format(method,key,fold))
            all_idx.extend(idx[:feature_num])
            #all_idx.extend(get_most_important_features(os.path.join(method,"{}_feature_importance_{}.txt".format(method,key)),idx=True)[:feature_num])      
    elif cluster_method=="overlap":
        for key in labels_dict.keys():
            feature_dict = find_over_lap_features(method,key,True)
            for k,v in feature_dict.items():
                all_idx.extend(v[:feature_num])

    all_idx = list(set(all_idx))

    score,all_idx_overlap = load_features("overlap_at_least_3.npy")
    intersection_idx = list(set(all_idx_overlap).intersection(set(all_idx)))
    for i in intersection_idx:
        print(feature_names[i])
    
    train_features,train_labels = util.load_features(prefix,output_dir,feature_name,fold,"train")
    train_features_selected=train_features[:,all_idx]
    test_features,test_labels = util.load_features(prefix,output_dir,feature_name,fold,"test")    
    test_features_selected = test_features[:,all_idx]
    print("original",test_features_selected.shape)
    lda = LDA(n_components=None, priors=None, shrinkage=None,
        solver='svd', store_covariance=False, tol=0.0001)
    test_features_selected = lda.fit_transform(test_features_selected,test_labels)
    print("lda",test_features_selected.shape)
    #test_features_selected=PCA(n_components=2).fit_transform(test_features_selected)
    tf= TSNE(n_components=2,perplexity=30).fit_transform(test_features_selected)
    colors=['r','darkgreen','y','c','b']
    curr_colors=np.asarray([colors[int(i)] for i in test_labels])

    selected_idx = get_num_samples(test_labels)
    selected_colors = curr_colors[selected_idx]
    plt.scatter(tf[selected_idx,0],tf[selected_idx,1],c=selected_colors,alpha=0.7)

    #plt.scatter(tf[:,0],tf[:,1],c=curr_colors,alpha=0.7)

    legend_elements = [Line2D([0], [0], marker='o', color='w', label='CRY',
                          markerfacecolor='r', markersize=10),
                    Line2D([0], [0], marker='o', color='w', label='FUS',
                          markerfacecolor='darkgreen', markersize=10),
                    Line2D([0], [0], marker='o', color='w', label='LAU',
                          markerfacecolor='y', markersize=10),
                    Line2D([0], [0], marker='o', color='w', label='BAB',
                          markerfacecolor='c', markersize=10),
                    Line2D([0], [0], marker='o', color='w', label='SCR',
                          markerfacecolor='b', markersize=10)]
    plt.legend(handles=legend_elements,loc="upper right")    
    #plt.show()
    #plt.savefig("overlap_at_least_3_reduced_bab_600_other_120.png")
    plt.savefig("{}/{}_top_{}_reduced_bab_600_other_120.png".format(method,cluster_method,feature_num))
    plt.close()

prefix="/home/jialu/disk1/infant-vocalize/"
output_dir="full_mode/merged/merged_idp_lena_5way/"
test_output_dir="full_mode/merged/merged_idp_lena_5way/"
target_folder = "lena_idp_5way"

# prefix="/media/jialu/Elements/"
# output_dir="CRIED_features/"
#target_folder = "CRIED"

labels_dict={"CRY":0,"FUS":1,"LAU":2,"BAB":3,"SCR":4}
idx2labels={0:"CRY",1:"FUS",2:"LAU",3:"BAB",4:"SCR"}
#feature_names=read_feature_name(os.path.join(prefix,"combined_feature_name_dur.txt"))

#PCA_analysis(method="fisher",cluster_method="one-vs-one")
# for method in ["fisher","extratree","mrmr","chi2"]:
#     print(method)
#     PCA_analysis(method=method,cluster_method="one-vs-one")

# fold=""
# feature_name = "embo_norm"
# train_features,train_labels = util.load_features(prefix,output_dir,feature_name,fold,"train")
# test_features,test_labels = util.load_features(prefix,output_dir,feature_name,fold,"test")
#compute_features(train_features,train_labels,"fisher_{}.npy".format(feature_name),method="fisher")
#load_feature_sets("fisher",feature_num=1000,padded=True)

# for j in [1,2,3]:
#     fold=j
#     feature_name="embo{}_norm".format(fold)
#     #train_features,_ = util.read_h5(os.path.join(prefix,output_dir,"train_selected_cnn_{}_norm.h5".format(feature_name)))
#     #train_labels,_ = util.read_h5(os.path.join(prefix,output_dir,"train_selected_cnn_{}_label.h5".format(feature_name)))
#     train_features,train_labels = util.load_features(prefix,output_dir,feature_name,fold,"train")
#     test_features,test_labels = util.load_features(prefix,test_output_dir,feature_name,fold,"test")
#     #compute_features(train_features,train_labels,"fisher_{}.npy".format(fold),method="fisher")
#     load_feature_sets("fisher",feature_num=1000,padded=True,save=False)

# num of feature vs acc/f1
#for i in [10,20,50,100,1000]:
for i in [1000]:
    accs, mf1s, wf1s = [],[],[]
    for j in [1,2,3]:
        fold=j
        feature_name="embo{}_norm".format(fold)
        #model_name="train_combined{}_multiple_norm".format(fold)

        train_features,train_labels = util.load_features(prefix,output_dir,feature_name,fold,"train")
        test_features,test_labels = util.load_features(prefix,output_dir,feature_name,fold,"test")
        acc, mf1, wf1, conf=load_feature_sets("fisher",i,save=False,padded=False)
        if j==1: confs = conf
        else: confs += conf
        accs.append(acc)
        mf1s.append(mf1)
        wf1s.append(wf1)
    print("mean acc", np.mean(accs),"std acc",np.std(accs))
    print("mean weighted F1 scores", np.mean(wf1s),"std weighted F1 scores",np.std(wf1s))
    print("mean macro F1 scores", np.mean(mf1s),"std macro F1 scores",np.std(mf1s))
    print("Composite confusion matrix")
    print(confs)

# one vs one classifiers
# for count,combo in enumerate(combinations(labels_dict.items(),2)):
#     class1,class2=combo[0][1],combo[1][1]
#     accs, mf1s, wf1s = [],[],[]
#     for j in [1,2,3]:
#         fold=j
#         feature_name="combined{}_multiple_norm".format(fold)
#         model_name="train_combined{}_multiple_norm".format(fold)
#         target_folder = "lena_idp_5way"
#         train_features,train_labels = util.load_features(prefix,output_dir,feature_name,fold,"train")
#         test_features,test_labels = util.load_features(prefix,output_dir,feature_name,fold,"test")
#         acc, mf1, wf1, conf=train_model_one_versus_one(class1,class2,100,"chi2")
#         if j==1: confs = conf
#         else: confs += conf
#         accs.append(acc)
#         mf1s.append(mf1)
#         wf1s.append(wf1)
#     print("mean acc", np.mean(accs),"std acc",np.std(accs))
#     print("mean weighted F1 scores", np.mean(wf1s),"std weighted F1 scores",np.std(wf1s))
#     print("mean macro F1 scores", np.mean(mf1s),"std macro F1 scores",np.std(mf1s))
#     print("Composite confusion matrix")
#     print(confs)

#one vs all classifier
# for c in range(5):
#     accs, mf1s, wf1s = [],[],[]
#     for j in [1,2,3]:
#         fold=j
#         feature_name="combined{}_multiple_norm".format(fold)
#         model_name="train_combined{}_multiple_norm".format(fold)
#         target_folder = "lena_idp_5way"
#         train_features,train_labels = util.load_features(prefix,output_dir,feature_name,fold,"train")
#         test_features,test_labels = util.load_features(prefix,output_dir,feature_name,fold,"test")
#         acc, mf1, wf1, conf=train_model_one_versus_all(c,100,"chi2")
#         if j==1: confs = conf
#         else: confs += conf
#         accs.append(acc)
#         mf1s.append(mf1)
#         wf1s.append(wf1)
#     print("mean acc", np.mean(accs),"std acc",np.std(accs))
#     print("mean weighted F1 scores", np.mean(wf1s),"std weighted F1 scores",np.std(wf1s))
#     print("mean macro F1 scores", np.mean(mf1s),"std macro F1 scores",np.std(mf1s))
#     print("Composite confusion matrix")
#     print(confs)


# for method in ["fisher","extratree","mrmr","chi2"]:
#     for count,combo in enumerate(combinations(labels_dict.items(),2)):
#         class1,class2=combo[0][1],combo[1][1]
#         f=open(os.path.join(method,"{}_feature_importance_{}_{}.txt".format(method,combo[0][0],combo[1][0])),"w")
#         content=[]
#         target_folder = "lena_idp_5way"
#         content+=output_feature_names("{}_{}_{}".format(method,combo[0][0],combo[1][0]),feature_num=50,method=method)
#         f.writelines(content)
#         f.close()

#     for key in labels_dict.keys():
#         f=open(os.path.join(method,"{}_feature_importance_{}.txt".format(method,key)),"w")
#         content=[]
#         target_folder = "lena_idp_5way"
#         content+=output_feature_names("{}_{}".format(method,key),feature_num=50,method=method)
#         f.writelines(content)
#         f.close()

# all_idxes=[]
# for key in labels_dict.keys():
#     all_dicts=[]
#     for method in ["fisher","extratree","mrmr","chi2"]:
#         #print(method)
#         all_dicts.append(find_over_lap_features(method,key,True))
#     # overlap for all 4 methods
#     # overlap_features=dict()
#     # for curr_dict in all_dicts:
#     #     for k,v in curr_dict.items():
#     #         if k not in overlap_features: overlap_features[k]=set(curr_dict[k])
#     #         else: overlap_features[k] = overlap_features[k].intersection(curr_dict[k])
#     # for k,values in overlap_features.items():
#     #     print(k)
#     #     for v in sorted(values):
#     #         print(v)
#     #     all_idxes+=list(values)
#     # all_idxes=list(set(all_idxes))
#     # all_idxes = [int(x) for x in all_idxes]
#     # print(all_idxes)
#     # np.save(os.path.join(target_folder,"overlap.npy"),all_idxes)

#     # overlap for at least two methods
#     overlap_features=dict()
#     for curr_dict in all_dicts:
#         for k,v in curr_dict.items():
#             if k not in overlap_features: overlap_features[k]=curr_dict[k]
#             else: overlap_features[k] += curr_dict[k]
#     print("target class",key)
#     for k,values in overlap_features.items():
#         print(k)
#         count = Counter(values)
#         at_least_two_methods_intersection = [k for k in count.keys() if count[k]>=3]
#         for v in at_least_two_methods_intersection:
#             print(v)
#         all_idxes+=list(at_least_two_methods_intersection)
# all_idxes=list(set(all_idxes))
# all_idxes = [int(x) for x in all_idxes]
# print(all_idxes,len(all_idxes))
# np.save(os.path.join(target_folder,"overlap_at_least_3.npy"),all_idxes)

# accs, mf1s, wf1s = [],[],[]
# for j in [1,2,3]:
#     fold=j
#     feature_name="combined{}_multiple_norm".format(fold)
#     model_name="train_combined{}_multiple_norm".format(fold)

#     train_features,train_labels = util.load_features(prefix,output_dir,feature_name,fold,"train")
#     test_features,test_labels = util.load_features(prefix,output_dir,feature_name,fold,"test")
#     _,idx=load_features("overlap_at_least_3.npy")
#     train_features_selected = train_features[:,idx]
#     test_features_selected = test_features[:,idx]
#     util.write_h5(train_features_selected,os.path.join(target_folder,"train_overlap_at_least_3_{}.h5".format(fold)))
#     util.write_h5(test_features_selected,os.path.join(target_folder,"test_overlap_at_least_3_{}.h5".format(fold)))

#     acc, mf1, wf1, conf=perform_LDA(train_features_selected,train_labels,test_features_selected,test_labels)
#     if j==1: confs = conf
#     else: confs += conf
#     accs.append(acc)
#     mf1s.append(mf1)
#     wf1s.append(wf1)
# print("mean acc", np.mean(accs),"std acc",np.std(accs))
# print("mean weighted F1 scores", np.mean(wf1s),"std weighted F1 scores",np.std(wf1s))
# print("mean macro F1 scores", np.mean(mf1s),"std macro F1 scores",np.std(mf1s))
# print("Composite confusion matrix")
# print(confs)

