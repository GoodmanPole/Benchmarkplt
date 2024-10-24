import math
from scipy.stats import skew
from sklearn.metrics import precision_score, recall_score, f1_score

class PerformanceMetrics:
    def __init__(self):
        pass

    def accuracy(self, actual, predicted):
        # Calculate accuracy percentage between test and train dataset
        correct = 0
        for i in range(len(actual)):
            if actual[i] == predicted[i]:
                correct += 1
        return correct / float(len(actual)) * 100.0


    def mae(self,actual,predicted):
        # MAE = sum( abs(predicted_i - actual_i) ) / total predictions

        sum_error = 0.0
        for i in range(len(actual)):
            sum_error += abs(predicted[i] - actual[i])
        return sum_error / float(len(actual))


    def mape(self,actual,predicted):
        # MAPE = (sum( abs(predicted_i - actual_i)  / actual value)) / n

        sum_error = 0.0
        for i in range(len(actual)):
            ape= abs((actual[i] - predicted[i])/actual[i])
            sum_error += ape
        return (sum_error / float(len(actual))) * 100



    def se(self,actual,predicted):
        # SE =  sum( (predicted_i - actual_i)^2 )
        sum_error = 0.0
        for i in range(len(actual)):
            prediction_error = predicted[i] - actual[i]
            sum_error += (prediction_error ** 2)
        squared_error = sum_error
        return squared_error


    def mse(self,actual,predicted):
        # MSE =  sum( (predicted_i - actual_i)^2 ) / total predictions
        sum_error = 0.0
        for i in range(len(actual)):
            prediction_error = predicted[i] - actual[i]
            sum_error += (prediction_error ** 2)
        mean_error = sum_error / float(len(actual))
        return mean_error


    def rmse(self,actual,predicted):
        # RMSE = sqrt( sum( (predicted_i - actual_i)^2 ) / total predictions)
        sum_error = 0.0
        for i in range(len(actual)):
            prediction_error = predicted[i] - actual[i]
            sum_error += (prediction_error ** 2)
        mean_error = sum_error / float(len(actual))
        return math.sqrt(mean_error)


    def sd(self,data):
        # Standard Deviation
        n = len(data)
        mean = sum(data) / n
        variance= sum((x - mean) ** 2 for x in data) / (n)
        return math.sqrt(variance)


    def rsquared(self,actual,predicted):
        # R^2= 1- (sum((actual-predicted)^2) / sum((actual- mean)^2))

        sum=0.0
        sum_e1_squared = 0.0
        sum_e2_squared = 0.0

        # Calculating the Mean
        for a in range(len(actual)):
           sum += actual[a]

        mean = sum / len(actual)

        for i in range(len(actual)):
            e1_squared = ((actual[i]-predicted[i])**2)
            sum_e1_squared += e1_squared

            e2_squared = ((actual[i]-mean)**2)
            sum_e2_squared += e2_squared

        r_squared = 1 - (sum_e1_squared / sum_e2_squared)
        return r_squared


    def precision(self,actual,predicted):

        # Precision Score = TP / (FP + TP)

        return precision_score(actual,predicted)

    def recall(self,actual,predicted):

        # Recall Score = TP / (FN + TP)

        return recall_score(actual,predicted)



    def f1score(self,actual,predicted):

        # F1 Score = 2* Precision Score * Recall Score/ (Precision Score + Recall Score/)

        return f1_score(actual,predicted)

    def skewness(self,data):

        return skew(data)





