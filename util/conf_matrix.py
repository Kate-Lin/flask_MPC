import matplotlib.pyplot as plt
import numpy as np

def draw_conf_matrix(conf_matrix,tag,addr,title='confusion_matrix for UCI heart disease'):
    plt.clf()
    plt.cla()
    plt.imshow(conf_matrix, interpolation='nearest', cmap=plt.cm.Oranges)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(tag))
    plt.xticks(tick_marks, tag, rotation=-45)
    plt.yticks(tick_marks, tag)
    thresh = conf_matrix.max() / 2
    iters = np.reshape([[[i, j] for j in range(len(tag))] for i in range(len(tag))], (conf_matrix.size, 2))
    for i, j in iters:
        plt.text(j, i, format(conf_matrix[i, j]),color="white" if conf_matrix[i,j] > thresh else "black",size=15)  # 显示对应的数字
    plt.ylabel('Real label')
    plt.xlabel('Prediction')
    plt.tight_layout()
    plt.savefig(addr, bbox_inches='tight',format='svg')
