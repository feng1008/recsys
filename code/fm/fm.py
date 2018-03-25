import numpy as np
import tensorflow as tf
import pandas as pd
from config import *
from sklearn.model_selection import train_test_split 
from sklearn.preprocessing import OneHotEncoder
import pickle


class FM(object):
	"""docstring for FM"""
	def __init__(self, n_epoch = 200, batch_size = 256, learning_rate = 0.01, hidden_factor = 8, method = 'adam', has_intersection = True, has_normal = True, lambda_w = 0.0001, lambda_v = 0.0001, 
		train_data = 'data/u1.base.test', test_data = 'data/u1.test.test', save_file = 'model/fm'):
        # bind params to class
		self.n_epoch = n_epoch
		self.batch_size = batch_size
		self.learning_rate = learning_rate
		self.hidden_factor = hidden_factor
		self.method = method
		self.has_intersection = has_intersection
		self.has_normal = has_normal
		self.lambda_w = lambda_w
		self.lambda_v = lambda_v

		self.train_data = train_data
		self.test_data = test_data
		self.save_file = save_file

		self.N_EPOCHS = 1000
		self.feature_dict = os.path.join(MODEL_PATH, 'user_item.dict')

		self.w0 = tf.Variable(tf.zeros([1]), name = 'w0')


	def load_data(self, file_name):
		enc = OneHotEncoder()

		cols = ['user', 'item', 'rating', 'timestamp']
		# import pdb;pdb.set_trace()
		df  = pd.read_csv(file_name, delimiter='\t', names = cols, dtype = 'S32')
		label = df[['rating']].astype('float')
		# data = pd.get_dummies(df[['user', 'item']]).astype('float')
		if not os.path.exists(self.feature_dict):
			one_hot_data = enc.fit_transform(df[['user', 'item']].values)
			pickle.dump(enc, open(self.feature_dict, 'wb'))
		else:
			output = open(self.feature_dict, 'rb')
			enc = pickle.load(open(self.feature_dict, 'rb'))
		# import pdb;pdb.set_trace()
		one_hot_data = enc.transform(df[['user', 'item']])
		return one_hot_data, label.values

	def transform_user_item(self, x):
		if not os.path.exists(self.feature_dict):
			print("dict file do not exists.")
			exit(1)
		else:
			output = open(self.feature_dict, 'rb')
			enc = pickle.load(open(self.feature_dict, 'rb'))

		data = pd.DataFrame(x, columns = ['user', 'item'], dtype = 'S32')
		result = enc.transform(data)
		# return enc.transform(x)
		return result
		
	def predict(self, x):
		y_hat = tf.add(self.w0, tf.reduce_sum(tf.multiply(x, self.w), 1, keep_dims = True))

		if self.has_intersection:
			interactions = tf.multiply(0.5, tf.reduce_sum(tf.subtract(
				tf.pow(tf.matmul(x, tf.transpose(self.v)), 2), tf.matmul(tf.pow(x, 2), tf.transpose(tf.pow(self.v, 2)))
			)))
			y_hat = tf.add(y_hat, interactions)
		return y_hat


	def loss_function(self, y, y_hat):
		loss = tf.reduce_sum(tf.pow(tf.subtract(y, y_hat), 2))

		if self.has_normal:
			l2_norm = tf.reduce_sum(tf.add(tf.multiply(tf.constant(self.lambda_w), tf.pow(self.w, 2)), tf.multiply(tf.constant(self.lambda_v), tf.pow(self.v, 2))))	
			loss = tf.add(loss, l2_norm)
		return loss


	def get_optimizer(self):
		if self.method == 'adagrad':
			return tf.train.AdagradOptimizer(learning_rate=self.learning_rate, initial_accumulator_value=1e-8)
		elif self.method == 'adam':
			return tf.train.AdamOptimizer(learning_rate=self.learning_rate)
		elif self.method == 'momentum':
			return tf.train.MomentumOptimizer(learning_rate=self.learning_rate, momentum=0.95)
		else:
			return tf.train.GradientDescentOptimizer(learning_rate=self.learning_rate)


	def build_model(self, x, y):
		# x = tf.placeholder('float', shape = [None, None])
		loss = self.loss_function(y, self.predict(x))

		optimizer = self.get_optimizer().minimize(loss)
		return optimizer

	def valid_loss(self, x, y):
		return tf.sqrt(tf.reduce_mean(tf.pow(tf.subtract(y, self.predict(x)), 2)))

	def train(self):
		# train_x, train_y = self.load_data(self.train_data)
		all_data_x, all_data_y = self.load_data(DATA_PATH + '/all.test')
		train_x, test_x, train_y, test_y = train_test_split(all_data_x, all_data_y, test_size = 0.3)
		n, p = all_data_x.shape
		# import pdb;pdb.set_trace()

		x = tf.placeholder('float', shape=[None, p])
		y = tf.placeholder('float', shape=[None, 1])

		self.w = tf.Variable(tf.zeros([p]), name = 'w')
		self.v = tf.Variable(tf.random_normal([self.hidden_factor, p], stddev=0.01), name = 'v')

		# y_hat = tf.Variable(tf.zeros([n, 1]))

		optimizer = self.build_model(x, y)
		# Launch the graph.
		# import pdb;pdb.set_trace()
		init = tf.global_variables_initializer()
		saver=tf.train.Saver([self.w0, self.w, self.v])
		with tf.Session() as sess:
			sess.run(init)
			for epoch in range(self.N_EPOCHS):
				if (epoch + 1 ) % 500 == 0:
					saver.save(sess, self.save_file + '_' + str(epoch + 1))
				# import pdb;pdb.set_trace()
				indices = np.arange(train_x.shape[0])
				np.random.shuffle(indices)
				t_x, t_y = train_x[indices[:self.batch_size]], train_y[indices[:self.batch_size]]
				sess.run(optimizer, feed_dict={x: t_x.toarray(), y: t_y})

			# print('MSE: ', sess.run(error, feed_dict={x: train_x, y: train_y}))
			# loss = tf.reduce_sum(tf.pow(tf.subtract(y, self.predict(x)), 2))
			loss = self.valid_loss(x, y)
			print('Loss (regularized error):', sess.run(loss, feed_dict={x: train_x.toarray(), y: train_y}))
			# print('Predictions:', sess.run(self.predict(x), feed_dict={x: train_x, y: train_y}))
			
			# print('Learnt weights:', sess.run(self.w, feed_dict={x: train_x, y: train_y}))
			# print('Learnt factors:', sess.run(self.v, feed_dict={x: train_x, y: train_y}))


	def test(self, model_file):
		# all_data_x, all_data_y = self.load_data(DATA_PATH + '/all.test')
		# train_x, test_x, train_y, test_y = train_test_split(all_data_x, all_data_y, test_size = 0.3)
		test_x, test_y = self.load_data(DATA_PATH + '/test.test')

		if not os.path.exists(self.feature_dict):
			print("feature dict does not exist.")
			return 
		else:
			output = open(self.feature_dict, 'rb')
			enc = pickle.load(open(self.feature_dict, 'rb'))

		# import pdb;pdb.set_trace()
		# x = tf.placeholder('float', shape=[None, enc.active_features_.shape[0]])
		x = tf.placeholder('float', shape=[None, test_x.shape[1]])
		y = tf.placeholder('float', shape=[None, 1])

		graph = tf.get_default_graph()
		# init = tf.global_variables_initializer()
		with tf.Session() as sess:
			sess.run(tf.global_variables_initializer())

			new_sess = tf.train.import_meta_graph(self.save_file + '_1000.meta')
			new_sess.restore(sess, self.save_file + '_1000')
			self.w0 = graph.get_tensor_by_name("w0:0")
			self.w = graph.get_tensor_by_name("w:0")
			self.v = graph.get_tensor_by_name("v:0")

			loss = self.valid_loss(x, y)
			print('Loss (regularized error):', sess.run(loss, feed_dict={x: test_x.toarray(), y: test_y}))

		# print('Predictions:', sess.run(self.predict(x), feed_dict={x: test_x, y: test_y}))

	def user_recommend(self, model_file, userid, itemid_list, top_k):
		result = []
		# import pdb;pdb.set_trace()
		# user_item_array = np.array([[].append([userid, item_id]) for item_id in itemid_list])
		user_item_array = np.vstack((np.tile(userid, len(itemid_list)), itemid_list)).T
		user_item_norm = self.transform_user_item(user_item_array)


		x = tf.placeholder('float', shape=[None, user_item_norm.shape[1]])
		y = tf.placeholder('float', shape=[None, 1])

		graph = tf.get_default_graph()
		# init = tf.global_variables_initializer()
		with tf.Session() as sess:
			sess.run(tf.global_variables_initializer())

			new_sess = tf.train.import_meta_graph(self.save_file + '_1000.meta')
			new_sess.restore(sess, self.save_file + '_1000')
			self.w0 = graph.get_tensor_by_name("w0:0")
			self.w = graph.get_tensor_by_name("w:0")
			self.v = graph.get_tensor_by_name("v:0")

			pred_y = sess.run(self.predict(x), feed_dict={x: user_item_norm.toarray()})
			# print('Predictions:', pred_y)
			# result.append(pred_y)
			# import pdb;pdb.set_trace()
			ind = np.argsort(pred_y, axis = 0)[-top_k:user_item_norm.shape[0]]
			return [x[0] + 1 for x in ind]

	def run(self):
		# self.train()
		self.test(os.path.join(MODEL_PATH, 'fm_500'))

		# self.user_recommend(os.path.join(MODEL_PATH, 'fm_500'), 10, [np.random.randint(100) for x in range(20)])



def main():
	fm = FM()
	fm.run()

if __name__ == '__main__':
	main()
