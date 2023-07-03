import csv
import tensorflow as tf
import pandas as pd
import numpy as np
import random
import os



def data_process(args):
    # process data

    train_data_directory = os.path.join(args.data_dir, args.dataset, args.dataset + '_train.csv')
    # valid_data_directory = os.path.join(args.data_dir, args.dataset, args.dataset + '_valid.csv')#用来校准的一个数据集
    # test_data_directory = os.path.join(args.data_dir, args.dataset, args.dataset + '_test.csv')



    #读取数据  此处可以将valid当作验证集 也就是说应该把train的一部分分出来当验证集
    args.train_seqs, train_student_num, train_max_skill_id, train_max_question_id, feature_answer_id = load_data(
        train_data_directory, args.field_size, args.max_step)
    #提取 训练集人数 训练学生数量 训练技巧最大值 训练集问题ID 回答0/1代表数字

    # args.test_seqs, test_student_num, test_max_skill_id, test_max_question_id, _ = load_data(test_data_directory,
    #                                                                                          args.field_size,
    #                                                                                          args.max_step)
    # args.valid_seqs = args.test_seqs
    q_matrix_directory = os.path.join(args.data_dir, args.dataset, 'q_matrix.csv')
    q_matrix = pd.read_csv(q_matrix_directory, header=None)


    # args.valid_seqs, valid_student_num, valid_max_skill_id, valid_max_question_id, _ = load_data(valid_data_directory, args.field_size, args.max_step)
    #提取验证集的信息


    # student_num = train_student_num + test_student_num #改完以后应该是train+valid+test
    args.skill_num = q_matrix.shape[1]
    args.question_num = q_matrix.shape[0]
    args.qs_num = args.skill_num+args.question_num#确定问题+技能数
    args.feature_answer_size = feature_answer_id + 1#确定回答的数字
    print(args.skill_num)
    print(args.question_num)

    accuracy_data_directory = os.path.join(args.data_dir, args.dataset, 'accuracy.csv')
    args.diff = pd.read_csv(accuracy_data_directory, header=None)
    # args.diff = np.ones([args.question_num,1])-np.array(args.diff, dtype=object)
    args.diff = np.array(args.diff, dtype=object)
    args.diff = np.reshape(args.diff[:, 0], [args.qs_num - args.skill_num, 1]).tolist()


    # #构建GAT邻接矩阵
    # array1 = np.eye(args.skill_num, args.skill_num)  # 左上角矩阵
    # array2 = np.eye(args.question_num, args.question_num)  # 右下角矩阵
    # df1 = pd.DataFrame(array1)
    # df2 = pd.DataFrame(array2)
    # adj_t = q_matrix.values.T  # 转置矩阵 右上
    # df3 = pd.DataFrame(adj_t)
    # args.adj = pd.concat([pd.concat([df1, df3], axis=1), pd.concat([q_matrix, df2], axis=1)], axis=0)
    # 生成了（68，68）的邻接对称矩阵 并且传入
    relation_directory = os.path.join(args.data_dir, args.dataset, 'relation.csv')
    F1 = np.array(pd.read_csv(relation_directory, header=None))
    # train_data = np.array(pd.read_csv(train_data_directory, header=None), dtype=object)
    # TP, FN, FP = tongji(train_data, args.question_num, args.qs_num)
    q_matrix = np.array(q_matrix,dtype=np.int)
    
    # qs_adj_list = build_q_list(q_matrix, args.question_num, args.skill_num,TP,FN,FP,args.yvzhi)
    #生成依赖矩阵
    question_adj_list = build_q_list(q_matrix, args.question_num, args.skill_num, F1, args.yvzhi)
    qs_adj_list1 = build_q_list1(q_matrix, args.question_num, args.skill_num)

    # qs_adj_list,interactions = build_adj_list(args.train_seqs,args.test_seqs,args.skill_matrix,args.qs_num)#[[neighbor skill/question] for all qs]]
    args.question_question_neighbors, _ = extract_qs_relations(question_adj_list, args.skill_num, args.qs_num,
                                                               args.question_question_neighbor_num,
                                                               args.skill_neighbor_num)
    args.question_neighbors1, args.skill_neighbors = extract_qs_relations(qs_adj_list1, args.skill_num, args.qs_num,
                                                                          args.question_neighbor_num,
                                                                          args.skill_neighbor_num)
    # print(args.question_neighbors.shape)#the first s_num rows are 0
    # print(args.skill_neighbors.shape)
    # exit()
    return args

def build_q_list(q_matrix, question_num, skill_num,F1,yvzhi):#生成对应关系列表
    qs_adj_list = []
    for i in range(skill_num):  # 确定技巧连接的问题
        adj_list = []
        for j in range(question_num):
            if q_matrix[j][i] == 1:
                adj_list.append(j+skill_num)
        qs_adj_list.append(adj_list)
    for i in range(question_num):  # 确定问题连接的技巧
        adj_list = []
        for j in range(skill_num):
            if q_matrix[i][j] == 1:
                adj_list.append(j)
        for k in range(question_num):
            if i != k:
                # if 2*(TP[i][k]/(TP[i][k]+FP[i][k]))*(TP[i][k]/(TP[i][k]+FN[i][k]))/((TP[i][k]/(TP[i][k]+FP[i][k]))+(TP[i][k]/(TP[i][k]+FN[i][k]))) >yvzhi:
                #     adj_list.append(skill_num + k)
                if (F1[i][k] + F1[k][i]) / 2 > yvzhi:
                    adj_list.append(skill_num + k)
        qs_adj_list.append(adj_list)

    return qs_adj_list

def build_q_list1(q_matrix, question_num, skill_num):#生成对应关系列表
    qs_adj_list = []
    for i in range(skill_num):  # 确定技巧连接的问题
        adj_list = []
        for j in range(question_num):
            if q_matrix[j][i] == 1:
                adj_list.append(j + skill_num)
        qs_adj_list.append(adj_list)
    for i in range(question_num):  # 确定问题连接的技巧
        adj_list = []
        for j in range(skill_num):
            if q_matrix[i][j] == 1:
                adj_list.append(j)
        qs_adj_list.append(adj_list)

    return qs_adj_list

def select_part_seqs(min_len, max_len, seqs):#选择适合做题数的学生
    temp_seqs = []
    for seq in seqs:
        if len(seq) >= min_len and len(seq) <= max_len:
            temp_seqs.append(seq)

    print("seq num is: %d" % len(temp_seqs))
    return temp_seqs




def extract_qs_relations(qs_list, s_num, qs_num, q_neighbor_size, s_neighbor_size):#建立问题技巧关系
    question_neighbors = np.zeros([qs_num, q_neighbor_size], dtype=np.int32)  # the first s_num rows are 0
    skill_neighbors = np.zeros([s_num, s_neighbor_size], dtype=np.int32)
    s_num_dic = {}
    q_num_dic = {}
    for index, neighbors in enumerate(qs_list):
        if index < s_num:  # index小于s_num则 代表s知识点
            if len(neighbors) not in q_num_dic:
                q_num_dic[len(neighbors)] = 1
            else:
                q_num_dic[len(neighbors)] += 1
            if len(neighbors) > 0:
                if len(neighbors) >= s_neighbor_size:
                    skill_neighbors[index] = np.random.choice(neighbors, s_neighbor_size, replace=False)#replace表示能不能取相同数字
                    #从neighnors抽取数字 组成s_neighbors_size大小的数组
                else:
                    skill_neighbors[index] = np.random.choice(neighbors, s_neighbor_size, replace=True)
        else:  # q
            # print(len(neighbors))
            if len(neighbors) not in s_num_dic:
                s_num_dic[len(neighbors)] = 1
            else:
                s_num_dic[len(neighbors)] += 1
            if len(neighbors) > 0:
                if len(neighbors) >= q_neighbor_size:
                    question_neighbors[index] = np.random.choice(neighbors, q_neighbor_size, replace=False)
                else:
                    question_neighbors[index] = np.random.choice(neighbors, q_neighbor_size, replace=True)

    # q_num_dic = sorted(q_num_dic.items(), key=lambda d: d[1])
    # print(s_num_dic)
    # print(q_num_dic)
    # exit()
    return question_neighbors, skill_neighbors

def tongji(train_data, question_num, qs):
    TP = np.ones([question_num,question_num])*0.001
    FN = np.ones([question_num, question_num]) * 0.001
    FP = np.ones([question_num, question_num]) * 0.001
    for i, index in enumerate(train_data):
        if i % 4 == 0:
            N = int(train_data[i][0])
            for k in range(0,N-1):
                for j in range(k+1,N):
                    if train_data[i+3][k] == qs and train_data[i+3][j] == qs+1:
                        FN[int(train_data[i+2][k] - (qs - question_num))][int(train_data[i+2][j] - (qs - question_num))] += 1
                    elif train_data[i+3][k] == qs+1 and train_data[i+3][j] == qs+1:
                        TP[int(train_data[i+2][k] - (qs - question_num))][int(train_data[i+2][j] - (qs - question_num))] += 1
                    elif train_data[i+3][k] == qs+1 and train_data[i+3][j] == qs:
                        FP[int(train_data[i+2][k] - (qs - question_num))][int(train_data[i+2][j] - (qs - question_num))] += 1
    return  TP, FN, FP

def load_data(dataset_path, field_size, max_seq_len):
    seqs = []
    student_id = 0
    max_skill = -1
    max_question = -1
    feature_answer_size = -1
    """
        第一行：作答习题数
        第二行：知识点序号
        第三行：习题序号
        第四行：表示习题答对答错，仅两种取值17904/17905
        field_size = 3
        max_seq_len = max_step = 200#应该改成50
        seqs: [
        [[skill1,problem1,correct1],[skill2,problem2,correct2],[...]...],
        [[skill1,problem1,correct1],[...]...]
        ...
        ]
        student_id: 学生数，一个作答序列组算是一个学生
        max_skill: 最大知识点id,也就是第二列中最大的数
        max_question: 最大问题id，也就是第三列中最大的数
        feature_answer_size: 17905
        """
    with open(dataset_path, 'r') as f:
        feature_answer_list = []
        for lineid, line in enumerate(f):
            fields = line.strip().strip(',')
            i = lineid % (field_size + 1)
            if i != 0:  # i==0 new student==>student seq len
                feature_answer_list.append(list(map(int, fields.split(","))))
            if i == 1:#找到每行最大值 获取到技巧数、问题数
                if max(feature_answer_list[-1]) > max_skill:
                    max_skill = max(feature_answer_list[-1])
            elif i == 2:
                if max(feature_answer_list[-1]) > max_question:
                    max_question = max(feature_answer_list[-1])
            elif i == field_size:
                student_id += 1
                if max(feature_answer_list[-1]) > feature_answer_size:# 数据最后两行表示答对答错，更新数据
                    feature_answer_size = max(feature_answer_list[-1])
                if len(feature_answer_list[0]) > max_seq_len:# feature_answer_list[0]也就是第一行，也就是skill,len()也就是时间步序列数,如果超过最大时间步就分片
                    n_split = len(feature_answer_list[0]) // max_seq_len # 先取整，得到几个max_step的序列
                    if len(feature_answer_list[0]) % max_seq_len:# 如果没有整除完，那么剩下的组成一个序列，分片数加一
                        n_split += 1
                else:
                    n_split = 1
                for k in range(n_split):# 对于每个分片，将其打包成[[skill1,problem1,correct1],[skill2,problem2,correct2],[...]...]
                    # Less than 'seq_len' element remained
                    if k == n_split - 1:# 学生序列数<=max_step时
                        end_index = len(feature_answer_list[0])# 就等于原序列数
                    else:
                        end_index = (k + 1) * max_seq_len# k是从0开始的，第0片就是第1(0+1)个分片，结束下标也就是max_step
                    split_list = []

                    for i in range(len(feature_answer_list)): # 对于每个分片分别处理三行数据，将其添加到分割列表中
                        split_list.append(feature_answer_list[i][k * max_seq_len:end_index])
                        # if i == len(feature_answer_list)-2:#before answer
                        # split_list.append([student_id]*(end_index-k*args.seq_len)) #student id

                    split_list = np.stack(split_list, 1).tolist()  # [seq_len,field_size]

                    seqs.append(split_list)
                feature_answer_list = []# 每处理完一个学生就清空处理下一个学生，每次只存放一个学生的序列

    return seqs, student_id, max_skill, max_question, feature_answer_size


def pad_sequences(sequences, maxlen=None, dtype='int32', padding='pre', truncating='pre', value=0.):
    """
       将每个批次不同长度的seq补成相同长度，后面补0 三元组全0
       :param sequences: 要补充的序列
       :param maxlen: 最后补成的长度
       :param dtype: 补的数据类型
       :param padding: 从前补还是从后面补
       :param truncating: 截断
       :param value: 以什么数值进行补充
       :return: 填补到相同长度的序列
       """
    lengths = [len(s) for s in sequences]
    nb_samples = len(sequences)
    if maxlen is None:
        maxlen = np.max(lengths)

    # take the sample shape from the first non empty sequence
    # checking for consistency in the main loop below.
    sample_shape = tuple()
    # print(np.shape(sequences))
    for s in sequences:
        if len(s) > 0:
            sample_shape = np.asarray(s).shape[1:]#获取一个三元组中元素数量
            break

    x = (np.ones((nb_samples, maxlen) + sample_shape) * value).astype(dtype)
            #初始化所有的答题
    for idx, s in enumerate(sequences):
        if len(s) == 0:
            continue  # empty list was found
        if truncating == 'pre':  # maxlen!=none may need to truncating
            trunc = s[-maxlen:]
        elif truncating == 'post':
            trunc = s[:maxlen + 1]
        else:
            raise ValueError('Truncating type "%s" not understood' % truncating)
        #pre从前面开始补0  post从后面开始补0

        # check `trunc` has expected shape
        trunc = np.asarray(trunc, dtype=dtype)
        if trunc.shape[1:] != sample_shape:
            raise ValueError('Shape of sample %s of sequence at position %s is different from expected shape %s' %
                             (trunc.shape[1:], idx, sample_shape))
        if padding == 'post':
            x[idx, :len(trunc)] = trunc
        elif padding == 'pre':
            x[idx, -len(trunc):] = trunc
        else:
            raise ValueError('Padding type "%s" not understood' % padding)
    return x


# select same skill index
def sample_hist_neighbors(seqs_size, max_step, hist_num, skill_index):
    # skill_index:[batch_size,max_step]
    """
        选择相关历史习题，硬选择
        :param seqs_size:
        :param max_step:
        :param hist_num:
        :param skill_index:
        :return:
        """
    # [batch_size,max_step,M]
    hist_neighbors_index = []

    for i in range(seqs_size):
        seq_hist_index = []
        seq_skill_index = skill_index[i]
        # [max_step,M]
        for j in range(1, max_step):
            same_skill_index = [k for k in range(j) if seq_skill_index[k] == seq_skill_index[j]]

            if hist_num != 0:
                # [0,j] select M
                if len(same_skill_index) >= hist_num:
                    seq_hist_index.append(np.random.choice(same_skill_index, hist_num, replace=False))
                else:
                    if len(same_skill_index) != 0:
                        seq_hist_index.append(np.random.choice(same_skill_index, hist_num, replace=True))
                    else:
                        seq_hist_index.append(([max_step - 1 for _ in range(hist_num)]))
            else:
                seq_hist_index.append([])
        hist_neighbors_index.append(seq_hist_index)
    return hist_neighbors_index


def format_data(seqs, max_step, feature_size, hist_num):
    """
        数据格式化
        :param seqs: 原始的练习数据序列
        :param max_step: 最大时间步200
        :param feature_size: 17904
        :param hist_num: 0     arg_parser.add_argument('--hist_neighbor_num', type=int, default=0)  # history neighbor num
        :return:[batch_size,max_len,feature_size]维度的 features_answer_index：32个学生，每个学生200个时间步的3个数据
                target_answers：目标回答，用最后一个特征值-17904 = 0/1
                seq_lens：32个学生练习序列未pad之前的最原始的长度
                hist_neighbor_index：历史习题的下标
        将每一个学生的做题序列补成最大长度  从后面开始补0
        """
    seqs = seqs
    seq_lens = np.array(list(map(lambda seq: len(seq), seqs)))

    # [batch_size,max_len,feature_size] 如[32,200,3]，其中的3表示知识点，习题，作答
    features_answer_index = pad_sequences(seqs, maxlen=max_step, padding='post', value=0)#从后面开始补0 补到最大步数
    answer_matrix = np.array([[j[-1] - feature_size for j in i[1:]] for i in seqs],dtype=object)
    target_answers = pad_sequences(answer_matrix, maxlen=max_step - 1, padding='post', value=0)
    #i代表 个人答题记录 j代表 某题答题记录 把除了第一题的答题记录做成target_answers
    """
    此处创建target_answer 调整loss
        """

    skills_index = features_answer_index[:, :, 0]   #提取技巧项
    hist_neighbor_index = sample_hist_neighbors(len(seqs), max_step, hist_num, skills_index)  # [batch_size,max_step,M]

    return features_answer_index, target_answers, seq_lens, hist_neighbor_index


class DataGenerator(object):

    def __init__(self, seqs, max_step, batch_size, feature_size, hist_num):  # feature_dkt
        np.random.seed(42)
        self.seqs = seqs
        self.max_step = max_step
        self.batch_size = batch_size
        self.batch_i = 0
        self.end = False
        self.feature_size = feature_size
        self.n_batch = int(np.ceil(len(seqs) / batch_size))
        self.hist_num = hist_num

    def next_batch(self):
        batch_seqs = self.seqs[self.batch_i * self.batch_size:(self.batch_i + 1) * self.batch_size]
        self.batch_i += 1

        if self.batch_i == self.n_batch:
            self.end = True

        format_data_list = format_data(batch_seqs, self.max_step, self.feature_size,
                                       self.hist_num)  # [feature_index,target_answers,sequences_lens,hist_neighbor_index]

        #此处格式化了数据 也就是说补全做题序列到最大步数为0
        return format_data_list


    def shuffle(self):
        self.pos = 0
        self.end = False
        np.random.shuffle(self.seqs)#打乱序列

    def reset(self):
        self.pos = 0
        self.end = False

# if __name__ == "__main__":
#     seqs = [[[1,0],[2,0],[3,1]],[[2,0],[4,1]]]
#     input_x, target_id, target_correctness, seq_len, max_len = format_data(seqs,1,5)
#     print(input_x)
#     print(target_id)
