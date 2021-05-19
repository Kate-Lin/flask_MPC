import matplotlib.pyplot as plt


def draw_ROC_curve(model_name,title,false_positive_rate, true_positive_rate,addr):
    plt.figure(figsize=(10, 5))
    plt.title(title)
    plt.plot(false_positive_rate, true_positive_rate, label=model_name)
    plt.plot([0, 1], ls='--')
    plt.plot([0, 0], [1, 0], c='.5')
    plt.plot([1, 1], c='.5')
    plt.ylabel('True positive rate')
    plt.xlabel('False positive rate')
    plt.legend()
    plt.savefig(addr)
