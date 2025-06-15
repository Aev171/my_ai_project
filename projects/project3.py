#duygu tespiti v1 cümleden duygu
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report

durumlar={0:"üzgün",1:"neşe",2:"aşk",3:"sinir",4:"korku"}

train=pd.read_csv(r"C:\Users\ATA\OneDrive\Masaüstü\training.csv")
test=pd.read_csv( r"C:\Users\ATA\OneDrive\Masaüstü\test.csv")

vectorizer=CountVectorizer(max_features=2000)

x=vectorizer.fit_transform(train["text"])
y=train["label"]

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)

model=MultinomialNB()
model.fit(x_train,y_train)

y_pred = model.predict(x_test)
print( {accuracy_score(y_test, y_pred)})
print(f"\n{classification_report(y_test, y_pred)}")

def deneme(metin):
    sayisay_metin=vectorizer.transform([metin])
    tahmin=durumlar[int(model.predict(sayisay_metin))]

    return tahmin

print(deneme("she is disrespectfull "))