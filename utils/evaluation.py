import torch
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score,\
    precision_score, recall_score, f1_score
import numpy as np
from sklearn.metrics import multilabel_confusion_matrix

def sensitivityCalc(Predictions, Labels):
    MCM = multilabel_confusion_matrix(Labels, Predictions,
                                      sample_weight=None,
                                      labels=None, samplewise=None)
    # MCM此处是 5 * 2 * 2的混淆矩阵（ndarray格式），5表示的是5分类

    # 切片操作，获取每一个类别各自的 tn, fp, tp, fn
    tn_sum = MCM[:, 0, 0] # True Negative
    fp_sum = MCM[:, 0, 1] # False Positive

    tp_sum = MCM[:, 1, 1] # True Positive
    fn_sum = MCM[:, 1, 0] # False Negative

    # 这里加1e-6，防止 0/0的情况计算得到nan，即tp_sum和fn_sum同时为0的情况
    Condition_negative = tp_sum + fn_sum + 1e-6

    sensitivity = tp_sum / Condition_negative
    macro_sensitivity = np.average(sensitivity, weights=None)

    micro_sensitivity = np.sum(tp_sum) / np.sum(tp_sum+fn_sum)

    return macro_sensitivity, micro_sensitivity

def specificityCalc(Predictions, Labels):
    MCM = multilabel_confusion_matrix(Labels, Predictions,
                                      sample_weight=None,
                                      labels=None, samplewise=None)
    tn_sum = MCM[:, 0, 0]
    fp_sum = MCM[:, 0, 1]

    tp_sum = MCM[:, 1, 1]
    fn_sum = MCM[:, 1, 0]

    Condition_negative = tn_sum + fp_sum + 1e-6

    Specificity = tn_sum / Condition_negative
    macro_specificity = np.average(Specificity, weights=None)

    micro_specificity = np.sum(tn_sum) / np.sum(tn_sum+fp_sum)

    return macro_specificity, micro_specificity

class evaluate():

    def __init__(self,cls_num=3):
        self.cls_num = cls_num
        # the confusion matrix: target/predict
        self.metrics = torch.zeros((cls_num,cls_num))
        self.sum = 0
        self.preds = np.array([])
        self.tars = np.array([])

    def calculation(self,targets,predicts):
        preds = torch.argmax(predicts, dim=1)
        tars = torch.argmax(targets, dim=1)
        self.preds = np.append(self.preds,preds.cpu().numpy())
        self.tars = np.append(self.tars,tars.cpu().numpy())
        for tar,pred in zip(tars,preds):
            self.metrics[tar,pred] += 1
            self.sum += 1

    def eval(self):
        acc = accuracy_score(self.tars,self.preds)
        precision = precision_score(self.tars,self.preds,average='micro')
        recall = recall_score(self.tars,self.preds,average='micro')
        f1 = f1_score(self.tars,self.preds,average='micro')
        sensitivity = sensitivityCalc(self.preds, self.tars)
        specificity = specificityCalc(self.preds, self.tars)

        print(self.metrics/self.sum)
        print('sum',self.sum)
        print('acc:',acc)
        print('precision:',precision)
        print('recall:',recall)
        print('f1_score:',f1)
        print('sensitivity:',sensitivity)
        print('specificity:',specificity)

        return acc,precision,recall,f1

    def show(self,thresh=50):
        C = self.metrics.numpy()
        S = C.sum(1)

        M = torch.zeros((self.cls_num,self.cls_num))
        for i in range(len(C)):
            for j in range(len(C)):
                M[j,i] = C[j, i]/S[j]*100
                print(C[j, i])
        plt.matshow(M, cmap='Greens')
        for i in range(len(C)):
            for j in range(len(C)):
                plt.annotate('%.1f'%M[j,i]+'%', xy=(i, j), horizontalalignment='center', verticalalignment='center',color="white" if M[j,i] > thresh else "black")
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.yticks([0, 1, 2], ['PCa', 'BPH', 'Control'])
        plt.xticks([0,1,2], ['PCa','BPH','Control'])
        plt.rcParams['figure.figsize']=(6,6)
        plt.tight_layout(pad=0.1, h_pad=None, w_pad=None, rect=None)
        plt.show()
