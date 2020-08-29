import matplotlib.pyplot as plt
import os

FORWARD_FILE_DIR = "../pie_chart_forward"
BACKWARD_FILE_DIR = "../pie_chart_backward"

def save_pie_chart(value,labels,file_name,title):
    new_value = []
    new_labels = []

    for i in range(len(value)):
        if float(value[i])!=0.0:
            new_value.append(value[i])
            new_labels.append(labels[i])
    plt.title(title)
    plt.rcParams['axes.labelsize'] = 12
    plt.rcParams['lines.linewidth'] = 2
    plt.rcParams['xtick.labelsize'] = 12
    plt.rcParams['ytick.labelsize'] = 12
    plt.rcParams['legend.fontsize'] = 12
    plt.rcParams['figure.figsize'] = [16,12]
    plt.axis('equal')
    plt.pie(new_value,labels=new_labels,autopct='%1.2f%%')
    plt.savefig(file_name)
    plt.clf()

def get_data(forward_file,backward_file):
    fw_list = []
    bw_list = []
    with open(backward_file,'r') as bw_file:
        for line in bw_file:
            tmp_list = []
            tmp_dict = {}
            line = line.strip().replace('\n','')
            tmp_list = line.split(' ')
            for i in range(0,len(tmp_list)):
                tmp_split = tmp_list[i].split(':')
                tmp_dict[tmp_split[0]] = tmp_split[1]
            bw_list.append(tmp_dict)
    #print(bw_list)

    bw_length = len(bw_list)
    tmp_fw_list = []
    with open(forward_file,'r') as fw_file:
        for line in fw_file:
            line = line.strip().replace('\n','')
            tmp_fw_list.append(line)

    del tmp_fw_list[-1]
    count = 0
    for i in range(0,len(tmp_fw_list)):
        if count==bw_length:
            break
        line = tmp_fw_list[len(tmp_fw_list)-1-i]
        tmp_list = []
        tmp_dict = {}
        tmp_list = line.split(' ')
        for i in range(0,len(tmp_list)):
            tmp_split = tmp_list[i].split(':')
            tmp_dict[tmp_split[0]] = tmp_split[1]
        fw_list.insert(0,tmp_dict)
        count = count + 1
    #print(fw_list)
    return fw_list,bw_list

if __name__ == '__main__':
    fw_list = []
    bw_list = []
    fw_list,bw_list = get_data(FORWARD_FILE_DIR,BACKWARD_FILE_DIR)

    output_path = './pie_chart'
    isExit = os.path.exists(output_path)
    if not isExit:
        os.makedirs(output_path)

    for iter in range(len(fw_list)):
        data = fw_list[iter]
        label = list(data.keys())
        value = list(data.values())
        save_pie_chart(value,label,"./pie_chart/fw_iteration_"+str(iter)+".png","Forward")

    for iter in range(len(bw_list)):
        data = bw_list[iter]
        label = list(data.keys())
        value = list(data.values())
        save_pie_chart(value,label,"./pie_chart/bw_iteration_"+str(iter)+".png","Backward")
    