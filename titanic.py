import pandas as pd

# Load Titanic dataset
url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
df = pd.read_csv(url)

print("FIRST 5 ROWS OF DATA")
print(df.head())

print("\nTOTAL PASSENGERS:", len(df))

# Count survived vs not survived
print("\nSURVIVAL COUNTS:")
print(df["Survived"].value_counts())

# Survival percentage
print("\nSURVIVAL PERCENTAGE:")
print(df["Survived"].value_counts(normalize=True) * 100)

# Survival by gender
print("\nSURVIVAL % BY GENDER:")
print(df.groupby("Sex")["Survived"].mean() * 100)

# Survival by class
print("\nSURVIVAL % BY CLASS:")
print(df.groupby("Pclass")["Survived"].mean() * 100)

# Average age by survival
print("\nAVERAGE AGE BY SURVIVAL:")
print(df.groupby("Survived")["Age"].mean())

# Children survival
children = df[df["Age"] < 18]
print("\nCHILDREN SURVIVAL COUNTS:")
print(children["Survived"].value_counts())

# Youngest & oldest passenger
print("\nYOUNGEST PASSENGER AGE:", df["Age"].min())
print("OLDEST PASSENGER AGE:", df["Age"].max())

# Family size
df["FamilySize"] = df["SibSp"] + df["Parch"]
alone = df[df["FamilySize"] == 0]
print("\nPEOPLE WHO TRAVELED ALONE:", len(alone))

print("\n--- ANALYSIS COMPLETE ---")

