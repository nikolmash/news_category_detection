from flask import Flask, request, render_template, redirect, url_for
from preprocess import TextPreprocessing

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
        #print(text)
        preprocess = TextPreprocessing(stop_words, lemmatize)
        preprocessed_text = preprocess.preprocess(text)
        print(preprocessed_text)
        result = 0  # здесь определяется класс
        with open('result.txt', 'w+') as f:
            f.write(str(result))

        return redirect(url_for('results'))

    return render_template('search.html')


@app.route('/results',  methods=['POST', 'GET'])
def results():
    with open('result.txt', 'r') as f:
        result = f.read()

    if request.method == "POST":
        print(request.form.getlist('response'))
        return render_template('statistics.html')

    return render_template('results.html', result=result)

@app.route('/statistics',  methods=['POST', 'GET'])
def statistics():
     return 'Done'


if __name__ == '__main__':
    app.run(debug=True)
