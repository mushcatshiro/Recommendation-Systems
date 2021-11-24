from models import I2ICF 

import os

import click
import pandas as pd

@click.command()
@click.option('--dataset')
@click.option('--fname')
@click.option('--drop')
def main(dataset, fname, drop):
    abs_path = os.path.join(os.getcwd(), 'dataset', dataset, fname)
    if not os.path.isfile(abs_path):
        raise FileNotFoundError
    df = pd.read_csv(abs_path)
    if drop:
        df.drop(drop, axis=1, inplace=True)
    model = I2ICF(input_matrix=df)
    model.train(
        item_col_name='movieId', 
        user_col_name='userId',
        rating_col_name='rating'
    )

if __name__ == '__main__':
    main()