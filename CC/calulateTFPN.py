import numpy as np
if __name__ == '__main__':
    """
        一些常用评价指标的计算
    """
    V = np.load('../CC/mobile_attn_TFPN_V.npy', allow_pickle='TRUE').item()
    A = np.load('../CC/mobile_attn_TFPN_A.npy', allow_pickle='TRUE').item()
    TP=V['TP']
    TN=V['TN']
    FP=V['FP']
    FN=V['FN']
    print('**'*20)
    n = TP + TN + FN + FP
    po = (TP + TN) / n
    print(f"accuracy:{po}")
    print(f"Precision:{TP / (TP + FP)}\nRecall:{TP / (TP + FN)}\n"
          f"F1Score:{2 * TP / (2 * TP + FN + FP)}")
    # Kappa系数是一个用于一致性检验的指标，也可以用于衡量分类的效果。因为对于分类问题，所谓一致性就是模型预测结果和实际分类结果是否一致。
    # 每一类正确分类的样本数量之和除以总样本数，也就是总体分类精度 对角线
    pe = ((TP + FN) * (TP + FP) + (FP + TN) * (FN + TN)) / (n * n)
    print(f"Kappa:{(po - pe) / (1 - pe)}")

    TP = A['TP']
    TN = A['TN']
    FP = A['FP']
    FN = A['FN']
    print('**' * 20)
    n = TP + TN + FN + FP
    po = (TP + TN) / n
    print(f"accuracy:{po}")
    print(f"Precision:{TP / (TP + FP)}\nRecall:{TP / (TP + FN)}\n"
          f"F1Score:{2 * TP / (2 * TP + FN + FP)}")
    # Kappa系数是一个用于一致性检验的指标，也可以用于衡量分类的效果。因为对于分类问题，所谓一致性就是模型预测结果和实际分类结果是否一致。
    # 每一类正确分类的样本数量之和除以总样本数，也就是总体分类精度 对角线
    pe = ((TP + FN) * (TP + FP) + (FP + TN) * (FN + TN)) / (n * n)
    print(f"Kappa:{(po - pe) / (1 - pe)}")



