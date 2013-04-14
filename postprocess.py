class PostProcess():

    @classmethod
    def submission(self, ids, preds):
        name = 'submission2.csv'
        f = open(name, 'w')
        string = ''
        ids = ids[0::4]
        if len(ids) != len(preds):
            raise Exception("The number of IDs and the number of predictions are different")
        for index,i in enumerate(ids):
            string += str(i) + ',' + str(preds[index]) + "\n"
        f.write(str(string))
        f.close()
