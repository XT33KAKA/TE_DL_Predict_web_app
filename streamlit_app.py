import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from Bio import SeqIO
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import GenericUnivariateSelect, mutual_info_classif
from propythia.protein.sequence import ReadSequence
from propythia.protein.descriptors import ProteinDescritors

# 定义函数以预处理序列
def preprocess_sequences(file_path):
    """
    预处理FASTA文件中的蛋白质序列。
    替换无法确定的氨基酸符号，并返回包含ID、序列和标签的DataFrame。
    """
    test_new_data = []
    sequences = SeqIO.parse(file_path, "fasta")
    for record in sequences:
        test_new_data.append([record.id, str(record.seq), " "])
    nb = pd.DataFrame(test_new_data, columns=["ID", "sequence", 'label'])
    
    # 执行氨基酸替换
    read_seqs = ReadSequence()
    res = read_seqs.par_preprocessing(dataset=nb, col='sequence', B='N', Z='Q', U='C', O='K', J='I', X='')

    return res

# 定义函数以计算物理化学特征
def calculate_features(dataset):
    """
    计算蛋白质序列的物理化学特征，并进行特征选择。
    返回标准化和选择后的特征矩阵。
    """
    descriptors_res = ProteinDescritors(dataset=dataset, col='sequence')
    res = descriptors_res.get_all_physicochemical(ph=7, amide=False, n_jobs=4)
    res2 = res.loc[:, (res != 0).any()]
    res2 = res2.loc[:, ~res2.T.duplicated(keep='first')]
    X = res2.drop(columns=['ID', 'sequence', 'label'])
    y = res2['label']
    scaler = StandardScaler().fit(X)
    X = scaler.transform(X)
    transformer = GenericUnivariateSelect(mutual_info_classif, mode='percentile', param=50).fit(X, y)
    X = transformer.transform(X)
    return X

# 定义函数以进行预测
def predict(X, model):
    """
    使用预训练的模型对特征矩阵进行预测。
    返回预测结果。
    """
    predictions = model.predict(X)
    return predictions.flatten()

# Streamlit应用的主函数
def main():
    st.title("Mining microbial tolerance element based on Deep learning")
    st.write("Upload a FASTA file to predict whether a protein sequence is a microbial tolerance element")
    st.write("Developed by: Synthetic Biology Laboratory, Xinjiang University")

    # 文件上传器
    uploaded_file = st.file_uploader("上传FASTA文件", type=['fasta', 'fa'])

    if uploaded_file is not None:
        # 加载预训练模型
        model_path = 'models/protein_descriptors_DL_best_model.h5'  # 替换为保存模型的实际路径
        model = load_model(model_path)

        # 预处理序列
        preprocessed_data = preprocess_sequences(uploaded_file.name)
        
        # 计算物理化学特征
        X_features = calculate_features(preprocessed_data)
        
        # 进行预测
        predictions = predict(X_features, model)
        
        # 创建结果DataFrame
        results = pd.DataFrame({
            'ID': preprocessed_data['ID'].values,
            'Prediction': predictions,
            'Sequence': preprocessed_data['sequence'].values
        })
        
        # 显示结果
        st.write(results)

if __name__ == "__main__":
    main()
