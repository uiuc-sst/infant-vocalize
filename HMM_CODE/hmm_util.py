import pdb
import tgt
import csv
import numpy as np
import os
from sklearn.metrics import f1_score

def map(string_label):
    if string_label == 'CRY': return 0
    if string_label == 'FUS': return 1
    if string_label == 'LAU': return 2
    if string_label == 'BAB': return 3
    if string_label == 'HIC': return 4
    else: return

def getBmatrix_binary(TextGrid_Directory, TextGridname, CsvDirectory):
    tg = tgt.read_textgrid(TextGrid_Directory+TextGridname)
    ipu_tier = tg.get_tier_by_name('Key-Child')
    filename = TextGridname.split('.')[0]

    # dictionaries with key of fileanme, values of softmax probability predictions being that label
    prob_lau = {}
    with open(CsvDirectory+'lau_prob.csv') as csvfile:
        reader = csv.reader(csvfile, delimiter=' ', quotechar='|')
        # prob_filenames = [row[0].split(',')[0] for row in reader]
        for row in reader:
            if row[0].split(',')[0] == 'total_filename':
                continue
            prob_lau[row[0].split(',')[0]] = float(row[0].split(',')[-2])
    prob_cry = {}
    with open(CsvDirectory+'cry_prob.csv') as csvfile:
        reader = csv.reader(csvfile, delimiter=' ', quotechar='|')
        # prob_filenames = [row[0].split(',')[0] for row in reader]
        for row in reader:
            if row[0].split(',')[0] == 'total_filename':
                continue
            prob_cry[row[0].split(',')[0]] = float(row[0].split(',')[-2])
    prob_bab = {}
    with open(CsvDirectory+'bab_prob.csv') as csvfile:
        reader = csv.reader(csvfile, delimiter=' ', quotechar='|')
        # prob_filenames = [row[0].split(',')[0] for row in reader]
        for row in reader:
            if row[0].split(',')[0] == 'total_filename':
                continue
            prob_bab[row[0].split(',')[0]] = float(row[0].split(',')[-2])
    prob_fus = {}
    with open(CsvDirectory+'fus_prob.csv') as csvfile:
        reader = csv.reader(csvfile, delimiter=' ', quotechar='|')
        # prob_filenames = [row[0].split(',')[0] for row in reader]
        for row in reader:
            if row[0].split(',')[0] == 'total_filename':
                continue
            prob_fus[row[0].split(',')[0]] = float(row[0].split(',')[-2])
    # pdb.set_trace()

    y = []
    prob_lau_order = []
    prob_cry_order = []
    prob_fus_order = []
    prob_bab_order = []

    # append the probabilities of that segment being different classes in order
    for seg in ipu_tier:
        start_time = seg.start_time
        end_time = seg.end_time
        annotation = seg.text
        segment_filename = filename+'-'+str(start_time)+'-'+str(end_time)+'-'+annotation
        if annotation == 'HIC':
            continue
        if not ((segment_filename in prob_lau) and (segment_filename in prob_cry) and (segment_filename in prob_fus) and (segment_filename in prob_bab)):
            continue
        if (segment_filename in prob_lau) and (segment_filename in prob_cry) and (segment_filename in prob_fus) and (segment_filename in prob_bab):
            prob_lau_order.append(prob_lau[segment_filename])
            prob_cry_order.append(prob_cry[segment_filename])
            prob_fus_order.append(prob_fus[segment_filename])
            prob_bab_order.append(prob_bab[segment_filename])
        y.append(map(annotation))
    prob_lau_order = np.array(prob_lau_order)
    prob_cry_order = np.array(prob_cry_order)
    prob_fus_order = np.array(prob_fus_order)
    prob_bab_order = np.array(prob_bab_order)

    # normalize the probabilies across the classes
    prob_sum = prob_lau_order+prob_cry_order+prob_fus_order+prob_bab_order
    prob_lau_order = prob_lau_order/prob_sum
    prob_cry_order = prob_cry_order/prob_sum
    prob_fus_order = prob_fus_order/prob_sum
    prob_bab_order = prob_bab_order/prob_sum

    res = np.zeros((4, len(y)))
    res[0, :] = prob_cry_order
    res[1, :] = prob_fus_order
    res[2, :] = prob_lau_order
    res[3, :] = prob_bab_order

    prediction_orignal = np.argmax(res, 0)
    accuracy = sum(y == prediction_orignal) * 1.0/len(y)
    # print('original accuracy:',accuracy)
    return res, accuracy, y

def normalize(m):
    # Given a 2d matrix, return it normalized by row.
    row_sums = m.sum(axis=1)
    return m / row_sums[:, np.newaxis]

def getBmatrix_multi(num_label, TextGrid_Directory, TextGridname, CsvDirectory):
    tg = tgt.read_textgrid(TextGrid_Directory+TextGridname)
    ipu_tier = tg.get_tier_by_name('Key-Child')
    filename = TextGridname.split('.')[0]

    # dictionaries with key of filename, values of softmax probability predictions being that label
    prob = {}
    with open(CsvDirectory) as csvfile:
        reader = csv.reader(csvfile, delimiter=' ', quotechar='|')
        # prob_filenames = [row[0].split(',')[0] for row in reader]
        for row in reader:
            if row[0].split(',')[0] == 'total_filename':
                continue
            if num_label == 4:
                prob[row[0].split(',')[0]] = [float(row[0].split(',')[-4]), float(row[0].split(',')[-3]), float(row[0].split(',')[-2]), float(row[0].split(',')[-1])]
            else:
                prob[row[0].split(',')[0]] = [float(row[0].split(',')[-5]), float(row[0].split(',')[-4]), float(row[0].split(',')[-3]), float(row[0].split(',')[-2]), float(row[0].split(',')[-1])]

    y = []
    prob_order = []
    # append the probabilities of that segment being different classes in order
    for seg in ipu_tier:
        start_time = seg.start_time
        end_time = seg.end_time
        annotation = seg.text
        segment_filename = filename + '-' + str(start_time) + '-' + str(end_time) + '-' + annotation
        if num_label == 4:
            if annotation == 'HIC':
                continue
        if not segment_filename in prob:
            continue
        if segment_filename in prob:
            prob_order.append(prob[segment_filename])
        y.append(map(annotation))

    prob_order = np.array(prob_order)
    res = prob_order.T
    prediction_orignal = np.argmax(res, 0)
    accuracy = sum(y == prediction_orignal) * 1.0/len(y)
    FSCORE = f1_score(y, prediction_orignal, average='macro')
    # print('original accuracy:',accuracy)
    return res, accuracy, y, FSCORE

def getAmatrix(TextGrid_Directory, CsvDirectory, N):
    A = np.zeros((N, N))
    # pdb.set_trace()
    for root, dirs, filenames in os.walk(TextGrid_Directory):
        # pdb.set_trace()
        for f in filenames:
            if f == '.DS_Store':
                continue
            tg = tgt.read_textgrid(TextGrid_Directory+f)
            ipu_tier = tg.get_tier_by_name('Key-Child')
            # pdb.set_trace()
            prev = map(ipu_tier[0].text)
            for i in range(1, len(ipu_tier)):
                cur = map(ipu_tier[i].text)
                if cur >= N:
                    break
                A[prev, cur] += 1
                prev = cur
    return A
