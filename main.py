import os
import math
import random
class Perceptron:
    def __init__(self, vecSize):
        self.weights = [random.uniform(-0.5, 0.5) for _ in range(vecSize)]
        self.t=3
        self.a=0.1
    def normalize(self, vector):
        norm = math.sqrt(sum(x * x for x in vector))
        return [x / norm for x in vector]

    def compute(self, vec):
        vec_normalized = self.normalize(vec)
        weights_normalized = self.normalize(self.weights)
        return (sum(w * v for w, v in zip(weights_normalized, vec_normalized))-self.t)

    def learn(self, vec, expected_output):
        actual_output = 1 if self.compute(vec) >= 0.5 else 0
        error = expected_output - actual_output
        learning_rate = self.a
        if error != 0:
            for i in range(len(self.weights)):
                self.weights[i] += learning_rate* error * vec[i]
                self.weights=self.normalize(self.weights)
            self.t=(self.t-(error)*learning_rate)
def createVecFromText(text):
    vector=[0]*26
    for char in text:
        if 'a'<=char<='z':
            vector[ord(char)-ord('a')]+=1

    return vector
class LanguageClassifier:
    def __init__(self, folder_paths):
        self.languages = list(folder_paths.keys())
        self.perceptrons = {lang: Perceptron(26) for lang in self.languages}
        self.train_perceptrons(folder_paths)

    def train_perceptrons(self, folder_paths):
        for lang, path in folder_paths.items():
            texts = self.load_texts_from_folder(path)
            vectors = [createVecFromText(text) for text in texts]
            for vec in vectors:
                for perceptron_lang, perceptron in self.perceptrons.items():
                    expected_output = 1 if perceptron_lang == lang else 0
                    perceptron.learn(vec, expected_output)
                    print(f"Learning {perceptron_lang}: Weights after update: {perceptron.weights}")
    def load_texts_from_folder(self, folder_path):
        import os
        texts = []
        for filename in os.listdir(folder_path):
            with open(os.path.join(folder_path, filename), 'r', encoding='utf-8') as file:
                text = file.read().encode('ascii', 'ignore').decode('ascii').lower()
                texts.append(text)
        return texts

    def classify(self, text):
        vec = createVecFromText(text)
        outputs = [perceptron.compute(vec) for perceptron in self.perceptrons.values()]
        return self.languages[max(range(len(outputs)), key=outputs.__getitem__)]

folder_paths = {
        "English": "LanguageToTrain/English",
        "French": "LanguageToTrain/French",
        "Polish": "LanguageToTrain/Polish"
    }

classifier = LanguageClassifier(folder_paths)

while True:
    text = input("Enter text to classify or 'quit' to exit: ")
    if text.lower() == 'quit':
        break
    language = classifier.classify(text)
    print(f"Classified as: {language}")