import pickle

data = []
with open('adult.pkl', 'rb') as file:
    while True:
        try:
            data.append(pickle.load(file))
        except EOFError:
            break

for item in data:
    print(item)
