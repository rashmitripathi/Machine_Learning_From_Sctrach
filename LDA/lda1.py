import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')


class LDA(object):
    def __init__(self, input_matrix):
        self.class_1 = np.ones(20)
        self.class_0 = np.zeros(20)
        self.targets = np.concatenate((self.class_0, self.class_1))
        data = np.column_stack((input_matrix, self.targets))
        self.input = np.copy(data)

    def get_eigenvalue_eigenvector(self, WStar):
        return np.linalg.eig(np.array(WStar))

    def perform_LDA(self):
        '''Performs LDA on the given input_matrix (ground truth is last column)'''
        x = self.input[:, :-1]
        # print(len(np.unique(self.input[:,-1])))
        total_classes = len(np.unique(self.input[:, -1]))
        mean_vector = []

        for k in range(0, total_classes):
            mu_for_class = np.sum(self.input[self.input[:, -1] == k], axis=0) / len(
                (self.input[self.input[:, -1] == k]))
            mean_vector.append(np.delete(mu_for_class, self.input.shape[1] - 1, 0))


            # print(mean_vector)
        # Calculate overall mean of the data..
        overall_mean = np.sum(x, axis=0) / len(x)
        # Calculate within class covariance matrix

        S_W = np.zeros((x.shape[1], x.shape[1]))
        for k in range(0, total_classes):
            xc = self.input[self.input[:, -1] == k][:, :-1]
            sc = np.zeros((x.shape[1], x.shape[1]))
            for m in range(0, xc.shape[0]):
                x_minus_mean = (xc[m] - mean_vector[k]).reshape(x.shape[1], 1)
                x_minus_mean_transpose = x_minus_mean.T
                sc += np.matmul(x_minus_mean, x_minus_mean_transpose)
            S_W += sc

        # Calculating S_between
        S_B = np.zeros((x.shape[1], x.shape[1]))
        for k in range(0, total_classes):
            mean_difference = (mean_vector[k] - overall_mean).reshape(overall_mean.shape[0], 1)
            mean_difference_T = mean_difference.T;
            S_B += np.matmul(mean_difference, mean_difference_T)

        W_star = np.matmul(np.linalg.inv(S_W), S_B)

        # Get EigenValues and EigenVectors
        eigen_values, eigen_vectors = self.get_eigenvalue_eigenvector(W_star)

        # Sorting
        indexes = eigen_values.argsort()[::-1]
        eigen_value = eigen_values[indexes]
        pcs = eigen_vectors[:, indexes]

        output = np.dot(x, pcs[:, 0:total_classes - 1])

        return mean_vector, S_W, S_B, W_star, output
#importing the data
df = pd.DataFrame.from_csv("SCLC_study_output_filtered_2.csv")
data_matrix = df.as_matrix()
lda_object = LDA(data_matrix) #Handled initialization!
mean,S_within,S_between, W_star, lda_output = lda_object.perform_LDA()


#Plotting the clusters from own LDA implementation
plt.figure(2)
plt.xticks([])
plt.yticks([])
plt.title("My LDA implementation")
plt.scatter(lda_output[0:20], [0]*20,c='orange')
plt.scatter(lda_output[20:40], [0]*20,c='green')
plt.show()