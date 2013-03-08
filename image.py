import re

data_file = "data/train.csv"
gender_file = "data/train_answers.csv"
env = 'train'

class Image():

	@classmethod
	def data(self):
		f = file(data_file, 'r')
		lines = f.readlines()[1:] # remove header
		lines = [re.sub("\r\n", '', line).split(',') for line in lines]
		floats = []
		for line in lines:
			del line[0] # remove ID
			del line[1] # remove language
			f = [float(i) for i in line]
			floats.append(f)
		return floats

	@classmethod
	def genders(self):
		f = file(gender_file, 'r')
		lines = f.readlines()[1:] # remove header
		lines = [re.sub("\r\n", '', line).split(',') for line in lines]
		quadrupled = []
		for line in lines:
			gender = [int(line[1])] * 4
			quadrupled += gender
		return quadrupled

#print len(Image.data()[0])
#print Image.genders()
