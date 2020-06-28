from flask import Flask, request, render_template, redirect, url_for
from preprocessing import TextPreprocessing
import sqlite3
from collections import Counter
import matplotlib.pyplot as plt
from get_models import get_model
import re

TOPICS = {0: 'Культура', 1: 'Мир', 2: 'Силовые структуры', 3: 'Наука и техника',
          4: 'Россия', 5: 'Спорт', 6: 'Дом', 7: 'Бывший СССР', 8: 'Экономика',
          9: 'Интернет и СМИ', 10: 'Из жизни', 11: 'Путешествия', 12: 'Ценности'}

app = Flask(__name__)


@app.route('/', methods=["GET", "POST"])
def search():
    if request.method == "POST":
        #print(request.form.getlist('mycheckbox'))
        selected = request.form.getlist('mycheckbox')
        if '1' in selected:
            stop_words = 1
        else:
            stop_words = 0
        if '2' in selected:
            lemmatize = 1
        else:
            lemmatize = 0
        #print(stop_words, lemmatize)
        text = str(request.form['text'])
        if re.search(r'[А-Яа-яЁё]', text) and len(text.split()) > 15:
            preprocess = TextPreprocessing(stop_words, lemmatize)
            preprocessed_text = preprocess.preprocess(text)
            tf_idf, model = get_model(stop_words, lemmatize)
            text_tf_idf = tf_idf.transform([preprocessed_text])
            result = TOPICS[model.predict(text_tf_idf)[0]]
        else:
            raise ValueError

        with open('result.txt', 'w+') as f:
            f.write(str(result))
        return redirect(url_for('results'))

    return render_template('search.html')


@app.route('/results',  methods=['POST', 'GET'])
def results():
    with open('result.txt', 'r') as f:
        result = f.read()
    if request.args:
        #print(request.args['response'])
        answer = request.args['response']
        conn = sqlite3.connect('data.db')
        cursor = conn.cursor()
        cursor.execute("""CREATE TABLE IF NOT EXISTS answers
                              (result, answer)
                           """)
        cursor.execute('INSERT INTO answers VALUES (?, ?)', (result, answer))
        conn.commit()
        return redirect(url_for('statistics'))

    return render_template('results.html', result=result)

@app.route('/statistics',  methods=['POST', 'GET'])
def statistics():
    conn = sqlite3.connect('data.db')
    cursor = conn.cursor()
    classes = []
    good_a = []
    for cl in cursor.execute('SELECT result, answer FROM answers').fetchall():
        classes.append(cl[0])
        if cl[1] == 'yes':
            good_a.append(cl[0])

    classes_count = Counter(classes)
    good_count = list(Counter(good_a).values())
    x = []
    for key in classes_count.keys():
        x.append(key)
    y = []
    classes_count = list(Counter(classes_count).values())
    for i in range(len(classes_count)):
        y.append(int(good_count[i])/int(classes_count[i]))
    #print(x, y)
    plt.title('Распределение правильных предсказаний по классам')
    plt.bar(x, y, color="salmon")
    plt.savefig('static/statistics.png')
    plt.clf()
    return render_template('statistics.html')


@app.route('/clouds')
def clouds():
    return render_template('clouds.html')


if __name__ == '__main__':
    app.run(debug=True)
