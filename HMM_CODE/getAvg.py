# coding: utf-8
# In[54]:
import sys
import json

# In[25]:
#filename = "e20170906_165204_011543_LC-4labels.txt"
filename = sys.argv[1]
Update_Accuracy = [-1 for i in range(5)]
Update_Fscore = [-1 for i in range(5)]
with open(filename) as f:
    lines = f.read().splitlines()

# In[31]:
NN_ACCURACY = 0
NN_FSCORE = 0
for idx in range(1,10,2):
    temp_line = lines[idx]
    line_elems = temp_line.split(" ")
    NN_ACCURACY = NN_ACCURACY + float(line_elems[7])
    NN_FSCORE = NN_FSCORE + float(line_elems[8])
avg_nn_accu = NN_ACCURACY / 5.0
avg_nn_fscore = NN_FSCORE / 5.0

# In[47]:
update_m = {}
#lambda:[accuracy, fsocre]
unique_lam = 0
for idx in range(1,len(lines),2):
    lam = lines[idx-1].split(" ")[1]
    temp_line = lines[idx].split(" ")
    if lam in update_m:
        update_m[lam] = [sum(x) for x in zip(update_m[lam], [float(temp_line[9]), float(temp_line[10])])]
    else:
        update_m[lam] = [float(temp_line[9]), float(temp_line[10])]
        unique_lam = unique_lam + 1

# In[52]:
avg_m = {}
for key, values in update_m.items():
    avg_m[key] = [x/float(unique_lam) for x in values]

# In[57]:
output = "output_"+filename
nn_accuracy = "Average NN Accuracy: " + str(avg_nn_accu) + "\n"
nn_fscore = "Average NN Fscore: " + str(avg_nn_fscore) + "\n"
with open(output, 'w') as file:
    file.write(nn_accuracy)
    file.write(nn_fscore)
    file.write(json.dumps(avg_m,sort_keys=True, indent=4, separators=(',', ': ')))
