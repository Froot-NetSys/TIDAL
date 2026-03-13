from collections import defaultdict

FILTER = 1000
FILE_MARKER = '<files>'
FILE_SPLIT = '/'
# FILE_SPLIT = '\\' # for Windows system

with open('object_data/cadet_object.txt', 'r') as file:
    input_ = file.read()
global total
global layer_dic


def attach(maindic, layer, ancestors):
    global layer_dic
    global path
    '''
    Insert a branch of directories on its trunk.
    # '''
    # print("layer = " + str(layer))
    if layer not in layer_dic:
        layer_dic[layer] = 1
    else:
        layer_dic[layer] += 1
    #     print(key)
    dic = {}
    for line in input_.split('\n'):
        line_info = line.split(' ')
        if len(line_info) != 2:
            continue
        line = line_info[0]
        num_of_line = int(line_info[1])
        branch = line
        parts = branch.split(FILE_SPLIT)

        a = ''

        if len(parts) > layer:
            for i in range(layer):
                a +=  parts[i] + FILE_SPLIT
            # a += '/'
            if a == ancestors:
                if parts[layer] not in dic:
                    dic[parts[layer]] = num_of_line
                    # print(parts[1])
                else:
                    dic[parts[layer]] += num_of_line
    end = False
    total_file = 0
    for key in dic:
        total_file += dic[key]
        count = 0
        if dic[key] > FILTER:
            count += 1
            a = ancestors + key + FILE_SPLIT
            maindic[key] = dict()
            attach(maindic[key], layer + 1, a)
        if count == 0:
            end = True
    if end:
        if ancestors not in path:
            path[ancestors] = total_file
        else:
            path[ancestors] += total_file

    return


def attach_no_filter(branch, trunk):
    '''
    Insert a branch of directories on its trunk.
    '''
    parts = branch.split(FILE_SPLIT, 1)
    if len(parts) == 1:  # branch is a file
        trunk[FILE_MARKER].append(parts[0])
    else:
        node, others = parts
        if node not in trunk:
            trunk[node] = defaultdict(dict, ((FILE_MARKER, []),))
        attach_no_filter(others, trunk[node])


def prettify(d, indent=0):
    '''
    Print the file tree structure with proper indentation.
    '''
    for key, value in d.items():
        if key == FILE_MARKER:
            if value:
                print('  ' * indent + str(value))
        else:
            print('  ' * indent + str(key))
            if isinstance(value, dict):
                prettify(value, indent + 1)
            else:
                print('  ' * (indent + 1) + str(value))


if __name__ == '__main__':
    # main_dict = defaultdict(dict, ((FILE_MARKER, []),))
    # for line in input_.split('\n'):
    #     attach_no_filter(line, main_dict)
    path = {}
    layer_dic = {}
    maindic = {}
    attach(maindic, 1, FILE_SPLIT)

    prettify(maindic)
    # for key in layer_dic:
    #     print("layer " + str(key) + ":" + str(layer_dic[key]))
    cc = 0
    for key in path:
        print(key)
        cc += 1
    print(cc)
