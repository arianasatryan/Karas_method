import os
import re
import json
train_path='/Users/hekpo/PycharmProjects/Karas/for_arm/data/train_dataset/'
test_path='/Users/hekpo/PycharmProjects/Karas/for_arm/data/test_dataset/'
train_files = os.listdir(train_path)
test_files = os.listdir(test_path)
train_results_path='/Users/hekpo/PycharmProjects/Karas/for_arm/results/train_results/'
test_results_path='/Users/hekpo/PycharmProjects/Karas/for_arm/results/test_results/'
saved_results_path=train_results_path


def get_starts_of_paragraphs(paragraphs):
    start_of_paragraph = {}
    i = 1
    length = 0
    for item in paragraphs:
        start_of_paragraph[i] = length
        i += 1
        length += (len(item)+1)
    return start_of_paragraph

def get_paragraphs_of(file_dir):
    with open(file_dir, encoding='utf-8')as fin:
            text=fin.read()
            # using ()just to save the real punctation mark,every even paragraph will contain only accured punctation-mark
            paragraphs = re.split('(:|։|\.|․)\n', text)
            for i in range(0, len(paragraphs) - 1, 2):  # adding missing splitt-punctation mark to the end of paragraph
                paragraphs[i] += paragraphs[i + 1]
            del paragraphs[1::2]            #deleting single even paragraphs with accured punctation-mark
    return paragraphs

def num_paragraphs_in(file_dir):
    return len(get_paragraphs_of(file_dir))

def avg_paragraph_len_in(file_dir):
    with open(file_dir,encoding='utf-8')as fin:
        text=fin.read()
        tokens=text.split()
    num_par=num_paragraphs_in(file_dir)
    return len(tokens)/num_par

def paragraphs_in_parts(file_dir,result_path):
   with open(file_dir,encoding='utf') as fin:
       text=fin.read()
   filename,file_format=os.path.splitext(os.path.basename(file_dir))
   with open(result_path + filename + '.truth', encoding='utf-8') as jsonfile:
        json_results = json.load(jsonfile)
        borders = json_results['positions']
        start_index = 0
        par = 1
        par_in_parts = {}
        for border in borders:
            white_spaces = re.findall('(:|։|\.|․)\n', text[start_index:border])
            par_in_parts[par] = len(white_spaces)
            start_index = border
            par += 1
        white_spaces = re.findall('(:|։|\.|․)\n', text[borders[-1]:])
        par_in_parts[par] = len(white_spaces) + 1
   return par_in_parts

def num_borders_in(file_dir,result_path):
    filename, file_format = os.path.splitext(os.path.basename(file_dir))
    with open(result_path + filename + '.truth', encoding='utf-8') as jsonfile:
        json_results = json.load(jsonfile)
        borders = json_results['positions']
    return len(borders)

def get_all_inform(files):
    if files==train_files:
        path=train_path
        train_or_test='train'
        result_path=train_results_path
    else:
        path=test_path
        train_or_test='test'
        result_path=test_results_path
    for file in files:
        filename, file_format = os.path.splitext(file)
        print('Filename:',filename)
        parag=get_paragraphs_of(path+file)
        num_par=num_paragraphs_in(path+file)
        max_num_bord=int(0.3 * num_par) + 1
        res_num_bord=num_borders_in(path+file,result_path)
        print('Number of paragraphs in this file:',num_par)
        print('Starts of the paragraphs:',get_starts_of_paragraphs(parag))
        print('Average paragraph length(by tokens):',avg_paragraph_len_in(path+file))
        print('Maximum allowable number of borders:',max_num_bord)
        print('Number of resulted borders:',res_num_bord)
        print('Procent of resulted borders from all:',res_num_bord*100/num_par)
        print('Number of paragraphs in resulted blocks:',paragraphs_in_parts(path+file,result_path))
        print('\n\n')

def compare_results(file_dir1,file_dir2,result_file_dir1,result_file_dir2):
    parags1=get_paragraphs_of(file_dir1)
    parags2=get_paragraphs_of(file_dir2)
    starts1=get_starts_of_paragraphs(parags1)
    starts2=get_starts_of_paragraphs(parags2)

    with open(result_file_dir1,encoding='utf-8')as json1f:
        json_results1 = json.load(json1f)
        borders1 = json_results1['positions']
        res1 = []
        for parnum, index in starts1.items():
            if index in borders1:
                res1.append(parnum)
    with open(result_file_dir2, encoding='utf-8')as json2f:
        json_results = json.load(json2f)
        borders2 = json_results['positions']
        res2 = []
        for parnum, index in starts2.items():
            if index in borders2:
                res2.append(parnum)
    comp_res={}
    comp_res['in res1 not in res2']=list(set(res1) - set(res2))
    comp_res['in res2 not in res1']=list(set(res2) - set(res1))
    print()
    return comp_res

def compare_feature_results_for(files):
    if files == train_files:
        path = train_path
        result_path = train_results_path
    else:
        path = test_path
        result_path = test_results_path
    for file in files:
        filename, file_format = os.path.splitext(file)
        file_dir1=path+filename+'.txt'
        result_file_dir1=result_path+filename+'.truth'
        result_file_dir2=saved_results_path+filename+'.truth'
        print(filename,'\n',compare_results(file_dir1,file_dir1,result_file_dir1,result_file_dir2),'\n')
