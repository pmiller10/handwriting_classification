class PostProcess():

    @classmethod
    def submission(self, ids, preds):
        name = 'submission1.csv'
        f = open(name, 'w')
        string = 'writer,male'
        for index,i in enumerate(ids):
            string += "\n" + str(i) + ',' + str(preds[index])
        f.write(str(string))
        f.close()
