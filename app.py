from flask import Flask, request, render_template, redirect, url_for
from app_functions import *
from pprint import pprint
import sys

app = Flask(__name__)


@app.route('/')
def home():
    return redirect(url_for('index'))


@app.route('/index')
def index():
    return render_template('index.html')

@app.route('/success/<name>')
def success(name):
    return 'welcome %s' % name


@app.route('/process', methods=['POST'])
def process():
    result = request.form
    name = result['Name']
    movie_list = get_recommended_movies(name)
    print(name)
    return render_template("success.html", result=movie_list)
    # title = request.form['title']
    # print(title)
    # sys.exit()
    # return render_template('success.html', result=result)
    #movie_list = get_recommended_movies(title)
    #return render_template('lists.html', error_code=error_code, imdbid_list=my_list, movie_list=movie_list)


if __name__ == '__main__':
    app.run(host='127.0.0.1', port=8080, debug=True)
