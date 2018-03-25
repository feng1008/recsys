import numpy as np
import tensorflow as tf
import pandas as pd
from config import *
from sklearn.model_selection import train_test_split 
from sklearn.preprocessing import OneHotEncoder
from fm_base import * 
import sys
import codecs


def read_set_from_file(input_file):
	result = set()
	for lines in codecs.open(input_file, 'r', 'utf-8'):
		result.add(lines.strip('\r\n'))
	return result

def read_dict_from_file(input_file):
	result = {}

	count = 0
	for lines in codecs.open(input_file, 'r', 'utf-8'):
		# import pdb;pdb.set_trace()
		try:
			item_no, item_name = lines.strip('\r\n').split('|')
		except:
			# import pdb;pdb.set_trace()
			count += 1
			continue
		result[item_no] = item_name
	return result


class Recommend(object):
	def __init__(self):
		self.user_list_file = os.path.join(DATA_PATH, 'user.list')
		self.item_list_file = os.path.join(DATA_PATH, 'item.list')
		self.item_name_dict_file = os.path.join(DATA_PATH, 'item_name.dict')

		self.num_of_recommend = 5
		self.num_of_filter_result = 100

		self.num_of_user_recommend = 10
		pass

	def random_filter(self, org_set):
		# filter_result = [list(org_set)[np.random.randint(len(self.item_list_file))] for i in range(self.num_of_filter_result)]
		filter_result = [list(org_set)[i] for i in [np.random.randint(0, len(self.item_list_file), self.num_of_filter_result)]]
		assert len(filter_result) > self.num_of_recommend
		return filter_result


	def recommend(self):
		user_set = read_set_from_file(self.user_list_file)
		# import pdb;pdb.set_trace()
		recommend_user_set = [list(user_set)[np.random.randint(len(user_set))] for i in range(self.num_of_user_recommend)]
		# item_set = read_set_from_file(self.item_list_file)

		item_name_dict = read_dict_from_file(self.item_name_dict_file)
		item_set = set(item_name_dict.keys())

		
		fm = FM()
		# import pdb;pdb.set_trace()
		for user_id in recommend_user_set:
			filter_result = self.random_filter(item_set)
			result = fm.user_recommend(os.path.join(MODEL_PATH, 'fm_500'), user_id, filter_result, self.num_of_recommend)
			# print("user:", user_id, "recommend item: ", ','.join([item_name_dict[str(x)] for x in result]))
			print("user:", user_id, ", recommend item: ", '|'.join([str(item_name_dict[str(x)]) for x in result]), "\n")

		

def main():
	# user_id = sys.argv[1]
	Recommend().recommend()
	pass

if __name__ == '__main__':
	main()
