# foldrpp

Classification of Heart Failure using Fold-R-PP 
On
 Heart Failure Prediction Dataset
https://www.kaggle.com/andrewmvd/heart-failure-clinical-data

About the Dataset:
Cardiovascular disease is the leading cause of death globally. It’s important to learn about your heart to help prevent it. If you have it, you can live a healthier, more active life by learning about your disease and taking care of yourself.
An estimated 17.9 million people died from CVDs in 2019, representing 32% of all global deaths. Of these deaths, 85% were due to heart attack and stroke. Over three quarters of CVD deaths take place in low- and middle-income countries.

Describing the Dataset:

The dataset contains 13 columns and 299 rows. The attributes are Age, Anaemia, Creatinine values, diabetes, ejection fraction, High Blood Pressure, Platelets count, serum creatinine, serum sodium, sex, smoking, Time and Death Event. All these factors help for the prediction of heart failure. Columns Anaemia, diabetes, high blood Pressure and smoking are Boolean type values which means ‘1’ indicates the presence and ‘0’ indicates the absence. The column Death event is the target column indicating whether a death occurs(indicated by 1) and 0 indicates that death has not occurred. There are NO missing values in the dataset.

Applying FOLD-R-PP Algorithm on the Dataset:

The FOLD-R++ algorithm takes tabular data as input, the first line for the tabular data should be the feature names of each column. The FOLD-R++ algorithm does not have to encode the data for training. It can deal with numerical, categorical, and even mixed type features (one column contains both categorical and numerical values) directly. However, the numerical features should be identified before loading the data, otherwise they would be dealt like categorical features (only literals with = and!= would be generated).








Fold-R-PP is applied on the above dataset using the following python code.

from foldrpp import *
from datasets import *
from timeit import default_timer as timer
from datetime import timedelta


def main():
    model, data = heart_failure()
    data_train, data_test = split_data(data, ratio=0.8, rand=True)  
    # line 28: 80% as training data, 20% as test data. shuffle data first when rand is True
    X_train, Y_train = split_xy(data_train)  # split data into features and label
    X_test,  Y_test = split_xy(data_test)

    start = timer()
    model.fit(X_train, Y_train, ratio=0.5)  
    # line 39: ratio means # of exception examples / # of default examples a rule can imply = 0.5  
    end = timer()
    model.print_asp(simple=True)
    Y_test_hat = model.predict(X_test)
    acc, p, r, f1 = get_scores(Y_test_hat, Y_test)
    print('% acc', round(acc, 4), 'p', round(p, 4), 'r', round(r, 4), 'f1', round(f1, 4))
    print('% foldr++ costs: ', timedelta(seconds=end - start), '\n')
    k = 1
    for i in range(14):
         print('Explanation for example number', k, ':')
         print(model.explain(X_test[i], all_flag=False))
         print('Proof Trees for example number', k, ':')
         print(model.proof(X_test[i], all_flag=False))
         k += 1
        
if __name__ == '__main__':
    main()

Rule Set:

After successful execution, the following rules set is generated and it is interpreted as follows;
% heart failure dataset 299 13
death_event(X,'1') :- time(X,N11), N11=<73.0, not ab3(X). 
Death occurs when the time of stroke is less than 73, and does not occurs when the ejection_fraction > 25, serum sodium not greater than 136 and less than 139, time not greater than 11, age not greater than 65 and creatinine not less than 75. Age not greater than is 65 years  is the bias in this rule since we cannot explain the reason the statement.
death_event(X,'1') :- creatinine_phosphokinase(X,N2), N2>59.0, ejection_fraction(X,N4), N4=<30.0, serum_creatinine(X,N7), N7>1.3, not ab4(X), not ab5(X). 
The second rule implies that death occurs when the creatinine is greater than 59, ejection < 30 , serum creatinine > 1.3, ejection fraction not greater than 20, age not less than 68, time not greater than 100 and less than 120.
ab1(X) :- creatinine_phosphokinase(X,N2), N2=<75.0. 
ab2(X) :- age(X,N0), N0>65.0, not ab1(X). 
ab3(X) :- ejection_fraction(X,N4), N4>25.0, serum_sodium(X,N8), N8>136.0, N8=<139.0, time(X,N11), N11>11.0, not ab2(X). 
ab4(X) :- anaemia(X,N1), N1=<0.0, ejection_fraction(X,N4), N4>20.0, sex(X,N9), N9>0.0. 
ab5(X) :- age(X,N0), N0=<68.0, time(X,N11), N11>100.0, N11=<120.0. 
% acc 0.8167 p 0.7647 r 0.65 f1 0.7027
The accuracy of the prediction is 81% and the F1 score for the model is 0.7
% foldr++ costs:  0:00:00.020927

Explanation of the Rule Set

The rules are explained using model.explain[X_test[i]] and the explanation is supported by the proof trees using model.proof[X_test[i]]
Explanation for example number 1 :
rebuttal 1:
[F]death_event(X,'1') :- [F]time(X,N11), N11=<73.0, not [U]ab3(X). 
{'time: 246.0'}
Since there is an observation with time > 73 and death event is 0; the above rule is termed as [F], where as [U] indicates that the rule is unnecessary for analysis.
rebuttal 2:
[F]death_event(X,'1') :- [T]creatinine_phosphokinase(X,N2), N2>59.0, [F]ejection_fraction(X,N4), N4=<30.0, [F]serum_creatinine(X,N7), N7>1.3, not [U]ab4(X), not [U]ab5(X). 
{'creatinine_phosphokinase: 582.0', 'serum_creatinine: 1.1', 'ejection_fraction: 38.0'}

Proof Trees for example number 1 :
rebuttal 1:
death_event(X,'1') DOES NOT HOLD because 
	the value of time is 246.0 which should be less equal to 73.0 (DOES NOT HOLD) 
{'time: 246.0'}
rebuttal 2:
death_event(X,'1') DOES NOT HOLD because 
	the value of creatinine_phosphokinase is 582.0 which should be greater than 59.0 (DOES HOLD) 
	the value of ejection_fraction is 38.0 which should be less equal to 30.0 (DOES NOT HOLD) 
	the value of serum_creatinine is 1.1 which should be greater than 1.3 (DOES NOT HOLD) 
{'creatinine_phosphokinase: 582.0', 'serum_creatinine: 1.1', 'ejection_fraction: 38.0'}

Insights

	If a high creatinine value is accompanied with high blood pressure and high serum creatinine value, death event occurs. This turns out to be a powerful implication of our analysis.
	If there is smoking with high blood pressure, there is high chance of death event due to heart failure. Thus, we can infer that a person with high blood pressure must quit smoking to save his heart

