from naiveBayesClassifier import tokenizer
from naiveBayesClassifier.trainer import Trainer
from naiveBayesClassifier.classifier import Classifier
import pandas as pd

def load_data():
    #Reading Dataset
    df=pd.read_csv('data/data.csv')
    return df
def age_group(age):
    if age<=2:
        return "baby"
    elif age>=3 and age<=39:
        return "young_adult"
    elif age>=40 and age<=59:
        return "middle_aged_adults"
    elif age>=60:
        return "old_adults"

def fit(df):
    #Trainer init
    model_trainer = Trainer(tokenizer)
    #Start training
    for row in df.iterrows():
        model_trainer.train(row[1]['x'],row[1]['y'])
    model= Classifier(model_trainer.data, tokenizer)
    return model

def predict(model,x,similar_count=2):
    disease=model.classify(x)
    return dict(enumerate([d[0] for d in  disease[:similar_count]]))

def questionnaire():
    print ("Please enter your Name:")
    name=input()
    print ("Please enter your Age:")
    age=float(input())
    print ("Please enter your Symptoms:")
    symptoms=input()
    return symptoms+" "+age_group(age),name
if __name__ == '__main__':
    df=load_data()
    model=fit(df)
    x,name=questionnaire()
    prediction =predict(model,x)
    print(f"Dear {name}, The closest disease:")
    print(prediction)
    
    
