from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

def MarvellousDecision(data_train,data_test,target_train, target_test):
    
    cobj = tree.DecisionTreeClassifier()
    
    cobj.fit(data_train,target_train)
    
    output = cobj.predict(data_test)
    
    Accuracy = accuracy_score(target_test, output)
    return Accuracy
    
def MarvellousKNN(data_train,data_test,target_train, target_test):
    
    
  
    
    cobj = KNeighborsClassifier()
    
    cobj.fit(data_train,target_train)
    
    output = cobj.predict(data_test)
    
    Accuracy = accuracy_score(target_test, output)
    return Accuracy
    
    
    
        
def main():

    dataset = load_iris()
    data = dataset.data
    target = dataset.target
    
    data_train,data_test,target_train, target_test= train_test_split(data,target,test_size = 0.5)
    

    ret=MarvellousDecision(data_train,data_test,target_train, target_test)
    print("Accurancy of dicisiion tree algorithm is : ",ret*100,"%" )

    ret=MarvellousKNN(data_train,data_test,target_train, target_test)
    print("Accurancy of KNN algorithm is : ",ret*100,"%" )


if __name__=="__main__":
    main()