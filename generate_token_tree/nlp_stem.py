import nltk
data = '../data/cadet/full_data/no_graph/processed_uuid_cadets.txt'


# # Enumerate the object
# with open( data ,'r' ) as fh:
#     all_object = []
#     for line in fh:
#         parts = line.split('\t')  # split line into parts
#         if len( parts ) > 1:  # if at least 2 parts/columns
#             all_object.append(parts[3])
#     #print(all_object)
# fh.close()
#
# freq = nltk.FreqDist(all_object)
# for key,val in freq.items():
#     print (str(key) + ' ' + str(val))

# Enumerate the action
with open( data ,'r' ) as fh:
    all_action = []
    for line in fh:
        parts = line.split('\t')  # split line into parts
        if len( parts ) > 1:  # if at least 2 parts/columns
            #if parts[2] not in all_event:
            all_action.append(parts[2])
    #print(all_action)
fh.close()

freq = nltk.FreqDist(all_action)
for key,val in freq.items():
     print (str(key))
