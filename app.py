from flask import Flask, render_template, request
import pickle
with open("model.pkl", "rb") as f:
    model, vectorizer = pickle.load(f)
app = Flask(__name__)
@app.route('/', methods=['GET', 'POST'])
def index():
    result = ''
    if request.method == 'POST':
        tweet = request.form['tweet']
        tweet_vec = vectorizer.transform([tweet])
        pred = model.predict(tweet_vec)[0]
        result = 'Cyberbullying Detected' if pred == 1 else 'No Cyberbullying'
    return '''
        <form method="post">
            <input name="tweet" placeholder="Enter tweet here">
            <input type="submit" value="Check">
        </form>
        <p>{}</p>
    '''.format(result)
if __name__ == '__main__':
    print("✅ Starting Flask server...")
    app.run(debug=True, port=5001)