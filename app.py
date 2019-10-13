from flask import Flask, request
from flask_cors import CORS
import pandas as pd
import warnings
import json
warnings.filterwarnings("ignore")

app = Flask(__name__)
CORS(app)


def get_df():
    csv_train = pd.read_csv('accidents.csv')
    csv_train = csv_train.drop(columns=['target'])
    csv_test = pd.read_csv('test.csv')
    csv = pd.concat([csv_train, csv_test])

    csv_predictions = pd.read_csv('merged_labels.txt')
    csv_predictions = csv_predictions.rename(columns={"prediction": "target"})

    csv = csv.merge(csv_predictions, on='accident_id')
    csv.set_index('accident_id', inplace=True)

    return csv


DATAFRAME = get_df()


def get_prob(df, list_accidents):
    df_accidents = df.loc[list_accidents, ['number_of_casualties', 'target']].copy()

    df_accidents['target'] = df_accidents['target'].copy().astype('float32')
    df_accidents['target'] = df_accidents['target'].copy() * df_accidents['number_of_casualties'].copy()

    return df_accidents['target'].sum() / df_accidents['number_of_casualties'].sum()


@app.route('/', methods=['GET', 'POST'])
def hello_world():
    accident_list = request.args.getlist('accident_id', type=int)

    accident_prob = get_prob(DATAFRAME, accident_list)
    return json.dumps({'prob': accident_prob})


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001)


