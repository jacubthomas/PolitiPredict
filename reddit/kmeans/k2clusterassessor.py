import numpy as np 
import pandas as pd 

clusters = 50
iterations = 20

'''VARIOUS FILES TO LOAD FROM'''
# df = pd.read_excel (f'D2V_BOW_20V_100E.xlsx')
# df = pd.read_excel (f'kmeans2_{clusters}_{iterations}iterations.xlsx')
df = pd.read_excel (f'kmeans2_{clusters}_{iterations}iterations_news.xlsx')


'''used this to evaluate which cluster belonged to which party and updated excel''' 
# cluster_counts_lib = [0 for i in range(clusters)]
# cluster_counts_con = [0 for i in range(clusters)]

# '''FOR EVERY ENTRY, READ PARTY AND ADD TO COUNT OF RESPECTIVE CLUSTER'''
# for index, row in df.iterrows():
#     if(row['party'] == 'Conservative'):
#         cluster_counts_con[row['cluster']] += 1
#     else:
#         cluster_counts_lib[row['cluster']] += 1

# '''SOME UPDATES FOR MYSELF'''
# print (cluster_counts_con)
# print ('\n\n\n')
# print (cluster_counts_lib)

# '''FIND LABELS FOR CLUSTERS.
#    WHICHEVER HAS MORE HITS, ASSIGN.'''
# l_count, c_count, ties = 0, 0, 0
# label_by_cluster = ['' for i in range(clusters)]
# for i in range(clusters):
#     if(cluster_counts_lib[i] > cluster_counts_con[i]):
#         label_by_cluster[i] = 'Liberal'
#         l_count += 1
#     elif(cluster_counts_lib[i] < cluster_counts_con[i]):
#         label_by_cluster[i] = 'Conservative'
#         c_count += 1
#     else:
#         label_by_cluster[i] = 'Tie'
#         ties += 1

# '''PUT LABELED CLUSTER PER ROW'''
# post_label_by_cluster = []
# for index, row in df.iterrows():
#     post_label_by_cluster.append (
#         label_by_cluster[row['cluster']]
#         )

# '''SOME UPDATES FOR MYSELF'''
# print (label_by_cluster)
# print (f'l_count: {l_count}, c_count: {c_count}, ties: {ties}')


# '''UPDATE THE SPREADSHEET'''
# df2=pd.DataFrame({'post_label_by_cluster': post_label_by_cluster})
# newdf = pd.concat([df, df2], axis=1)
# # newdf.to_excel(f'D2V_BOW_20V_100E.xlsx')
# # newdf.to_excel(f'kmeans2_db_{clusters}_{iterations}iterations.xlsx')
# newdf.to_excel(f'kmeans2_{clusters}_{iterations}iterations_news.xlsx')
'''used this to evaluate which cluster belonged to which party and updated excel''' 

correct, wrong, tie = 0, 0, 0

for index, row in df.iterrows():
    if (row['party'] == row[f'post_label_by_cluster']):
        correct += 1
    elif row[f'post_label_by_cluster'] != 'Tie':
        wrong += 1
    else:
        tie +=1
total = correct+wrong

print (f'accuracy = {correct/total}, with {correct} and {wrong} and {tie} ties!')

