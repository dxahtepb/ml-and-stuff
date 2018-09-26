class Metrics:
    def accuracy(test_data=[], predicted_data=[], *args):
        return sum([1 if test_data[i].label != predicted_data[i].label else 0 for i in range(len(test_data))]) / len(test_data)


    def f_score(test_data=[], predicted_data=[], n_classes=2, beta=1, *args):
        def get_c_matrix(test_data=[],predicted_data=[], n_classes=2):
            """
            lines represent classifier decisions
            columns represent test data
            """
            conf_matrix = [[0 for i in range(n_classes)] for i in range(n_classes)]
            for idx, test_case in enumerate(test_data):
                conf_matrix[predicted_data[idx].label][test_case.label] += 1
            return conf_matrix

        conf_matrix = get_c_matrix(test_data,predicted_data,n_classes)
        prec = 0
        rec = 0
        for i in range(n_classes):
            prec += conf_matrix[i][i] / sum(conf_matrix[i])
            rec += conf_matrix[i][i] / sum([conf_matrix[j][i] for j in range(len(conf_matrix))])
        prec = prec / n_classes
        rec = rec / n_classes
        res_score = (pow(beta, 2) + 1) * (prec * rec) / (pow(beta, 2) * prec + rec)
        return res_score