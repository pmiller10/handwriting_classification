import re

gender_file = "data/train_answers.csv"
env = 'train'

class Image():

    train_file = "data/train.csv"
    test_file = "data/test.csv"

    @classmethod
    def data(self, data_file=train_file):
        f = file(data_file, 'r')
        lines = f.readlines()[1:] # remove header
        lines = [re.sub("\r\n", '', line).split(',') for line in lines]
        floats = []
        ids = []
        for line in lines:
            ids.append(line[0])
            del line[0] # remove ID
            del line[1] # remove language
            page_type = self.page_type_to_vector(line[0])
            del line[0] # remove page type
            line = page_type + line
            f = [float(i) for i in line]
            floats.append(f)
        return floats, ids

    @classmethod
    def all(self):
        train, ids1 = self.data()
        test, ids2 = self.data(self.test_file)
        both = train + test
        ids = ids1 + ids2
        return both, ids

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

    @classmethod
    def page_type_to_vector(self, page_type):
        if page_type == '1':
            vect = [1,0,0,0]
        if page_type == '2':
            vect = [0,1,0,0]
        if page_type == '3':
            vect = [0,0,1,0]
        if page_type == '4':
            vect = [0,0,0,1]
        return vect

#print len(Image.data()[0])
#print Image.genders()
