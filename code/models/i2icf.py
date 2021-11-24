"""
pseudo algorithm

for each item in product catalog, I1
	for each customer C who purchased I1
		for each item I2 purchased by customer C
			record that a customer purchased I2
	for each item I2
		compute similarity between I1 and I2

similarity function: 
    1. cosine similarity
    2. frequency
score function: TBD
"""
import click
import pandas as pd


class I2ICF:
    def __init__(self, input_matrix:pd.DataFrame):
        """
        expected df:
        userId itemId rating
        1      1      5
        1      2      3
        2      1      5
        intermediate df:
        itemId itemId similarity_score
        1      2      0.1
        1      3      0.5
        result df:
        itemId itemId predicted_rating
        1      2      4
        1      3      1
        """
        self.input_matrix = input_matrix
        self.intermediate_matrix = pd.DataFrame(columns=['I1', 'I1S', 'I2', 'I2S'])

    def train(self, item_col_name:str, user_col_name:str, rating_col_name:str):
        items = self.input_matrix[item_col_name].unique()
        for i, target_item in enumerate(items):
            users_purchased_target_item = self.input_matrix[
                self.input_matrix[item_col_name] == target_item
            ][user_col_name]
            for user in users_purchased_target_item:
                user_all_items = self.input_matrix[
                    self.input_matrix[user_col_name] == user
                ]
                tmp = pd.DataFrame(columns=['I1', 'I1S', 'I2', 'I2S'])
                tmp['I2'] = user_all_items[
                    user_all_items[item_col_name] != target_item
                ][item_col_name]
                tmp['I2S'] = user_all_items[
                    user_all_items[item_col_name] != target_item
                ][rating_col_name]
                tmp['I1'] = target_item
                tmp['I1S'] = user_all_items[
                    user_all_items[item_col_name] == target_item
                ][rating_col_name].values[0]
                self.intermediate_matrix = self.intermediate_matrix.append(
                    tmp, ignore_index=True
                )
            break
        self.intermediate_matrix.to_csv('testout.csv')
        return self
