import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler,LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

df = pd.read_csv(r'houseprice\Housing.csv')
X=df.drop(columns=['price','prefarea'])
label=LabelEncoder()
for col in ['mainroad','guestroom','basement','airconditioning','hotwaterheating','furnishingstatus']:
    X[col]=label.fit_transform(X[col])
scaler = StandardScaler()
X = scaler.fit_transform(X)
y=df['price']
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)
lr=LinearRegression()
lr.fit(X_train,y_train)
y_pred= lr.predict(X_test)
score=r2_score(y_test,y_pred)

print(f"Model accuracy : {score:.3f}%")

def check():
    while True:
        try:
            n= int(input())
            return n
        except Exception:
            print("Enter valid input : ",end="")

l=[]
print("\n\n       Welcome to the HOME PRICE PREDICTION programme")
print("               ---------------------------           ")
l2=["Enter the estimated area of the house you want to buy (sq ft)",
    "How much bedrooms do you want?",
    "How much bathroom is preffered",
    "How many floors is needed",
    "Do you want the house near main road ? (yes: 1, no: 0)",
    "Do you want extra guestrooms (yes: 1, no: 0)?",
    "Is hot water heating is needed for your house (yes: 1, no: 0)?",
    "Do you want a seperated basement (yes: 1, no: 0)?",
    "Should your room be airconditioned ? (yes: 1, no: 0)",
    "How many cars do you have for parking ?",
    "What type of furnitures do you need ? (fursnished: 0, semi-furnished: 1, unfurnished: 2)"
    ]

for i in range(len(l2)):
    print(l2[i]," : ",end="")
    l.append(check())

l = np.array(l).reshape(1, -1)
print(l)
l = scaler.transform(l)

print(f"\nThe estimated price will be : {lr.predict(l)[0]:.2f}/-")