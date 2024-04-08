from flask import Flask, render_template, request
from chess_engine import *
from flask import jsonify
import pickle
# from dotenv import load_dotenv
import os
from pandas import read_csv
import google.generativeai as genai




# Load the .env file
# load_dotenv()


app = Flask(__name__)

genai.configure(api_key="<insert your api key>")
# Set up the model
generation_config = {
  "temperature": 0,
  "top_p": 1,
  "top_k": 1,
  "max_output_tokens": 2048,
}
 
safety_settings = [
  {
    "category": "HARM_CATEGORY_HARASSMENT",
    "threshold": "BLOCK_MEDIUM_AND_ABOVE"
  },
  {
    "category": "HARM_CATEGORY_HATE_SPEECH",
    "threshold": "BLOCK_MEDIUM_AND_ABOVE"
  },
  {
    "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
    "threshold": "BLOCK_MEDIUM_AND_ABOVE"
  },
  {
    "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
    "threshold": "BLOCK_MEDIUM_AND_ABOVE"
  }
]
 
model = genai.GenerativeModel(model_name="gemini-pro", generation_config=generation_config,safety_settings=safety_settings)




def make_model_data(arr, columns):
    res = []
    for i in range(0, len(columns)):
        res.append(0)
    for i in range(0, len(arr)):
        if(i==0):
            arr[0] = 'w1_'+arr[0]
        if(i==1):
            arr[1] = 'b1_'+arr[1]
        if(i==2):
            arr[2] = 'w2_'+arr[2]
        if(i==3):
            arr[3] = 'b2_'+arr[3]
        if(i==4):
            arr[4] = 'w3_'+arr[4]
        if(i==5):
            arr[5] = 'b3_'+arr[5]
        for j in range(0, len(columns)):
            if(columns[j]==arr[i]):
                res[j]=1
    return res

def make_predictions(wm, bm):
    df = read_csv("cols.csv")
    cols = list(df['0'])
    wmoves = ','.split(wm)
    bmoves = ','.split(bm)
    all_moves = []
    for i in range(0, len(wmoves)):
        all_moves.append(wmoves[i])
        all_moves.append(bmoves[i])
    res = make_model_data(all_moves, cols)
    res[0] = 0.000015
    res[1] = -0.000420
    loaded_model = pickle.load(open('CatBoost.pkl', 'rb'))
    result = loaded_model.score(res, 1)
    return int(result)

@app.route('/', methods=['GET', 'POST'])
def index():
    
    advantage = None # set a default value for advantage
    
    if request.method == 'POST':
        print("request is post successful")
        data = request.get_json()
        white_moves = data['white']        
        black_moves = data['black']
        print("Our Prediction:",make_predictions(white_moves, black_moves))
        print(white_moves)
        print(black_moves)
        prompt = f"Given the following chess moves:\nWhite: {white_moves}\nBlack: {black_moves}\nWho is currently in advantage and list all cells that this side have control on? What is the next suggested move for white, just write 1 code for the next move? Give in paragraph form"
        try:
            headers = {
                "Authorization": f"Bearer {openai.api_key}"
            }
            # response = openai.Completion.create(
            #     model="gpt-3.5-turbo-instruct",
            #     prompt=prompt,
            #     max_tokens=1024,
            # )
            response = model.generate_content(prompt)
            print(response.text)
            # print()
            # advantage = response.choices[0].text.strip()    
            advantage = response.text
            return jsonify({'advantage': advantage})

        except openai.error.InvalidRequestError as e:
            print(f"Invalid Request Error: {e}")
            return "An error occurred.invalid Please try again later."
        except openai.error.AuthenticationError as e:
            print(f"Authentication Error: {e}")
            return "An error occurred.authentication Please try again later."
        except openai.error.APIError as e:
            print(f"API Error: {e}")
            return "An error occurred. api  Please try again later."
        except Exception as e:
            print(f"Error: {e}")
            return "An error occurred.e Please try again later."
    print(advantage)        
    return render_template('index.html')

@app.route('/move/<int:depth>/<path:fen>/')
def get_move(depth, fen):
    print(depth)
    print("Calculating...")
    engine = Engine(fen)
    move = engine.iterative_deepening(depth - 1)
    print("Move found!", move)
    print()
    return move


@app.route('/test/<string:tester>')
def test_get(tester):
    return tester


if __name__ == '__main__':
    app.run(debug=True)
