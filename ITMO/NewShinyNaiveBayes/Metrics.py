class Metrics:
    @staticmethod
    def accuracy(test_data, predicted_data, *args):
        return sum([1 if test_data[i] == predicted_data[i] else 0 for i in range(len(test_data))]) / len(test_data)

    @staticmethod
    def f_score(test_data, predicted_data, n_classes=2, beta=1, *args):
        def get_c_matrix(test_data, predicted_data, n_classes=2):
            """
            lines represent classifier decisions
            columns represent test data
            """
            conf_matrix = [[0 for i in range(n_classes)] for i in range(n_classes)]
            for idx, test_case in enumerate(test_data):
                conf_matrix[predicted_data[idx]][test_case] += 1
            return conf_matrix

        conf_matrix = get_c_matrix(test_data, predicted_data, n_classes)
        prec = 0
        rec = 0

        for i in range(n_classes):
            div = sum(conf_matrix[i])
            if div == 0:
                prec += 1
            else:
                prec += conf_matrix[i][i] / div

            div = sum([conf_matrix[j][i] for j in range(len(conf_matrix))])
            if div == 0:
                rec += 1
            else:
                rec += conf_matrix[i][i] / div

        prec = prec / n_classes
        rec = rec / n_classes
        res_score = (pow(beta, 2) + 1) * (prec * rec) / (pow(beta, 2) * prec + rec)

        return res_score

    @staticmethod
    def f_score2(test_data, predicted_data):
        from sklearn.metrics import f1_score
        return f1_score(test_data, predicted_data)

    @staticmethod
    def get_confusion(test_data, predicted_data):
        from sklearn.metrics import confusion_matrix
        return confusion_matrix(test_data, predicted_data)

    @staticmethod
    def confusion_dict(test_data, predicted_data):
        x = Metrics.get_confusion(test_data, predicted_data)
        return {'TP': x[1][1], 'TN': x[0][0], 'FN': x[1][0], 'FP': x[0][1]}
