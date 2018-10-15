import numpy
import scipy.special
import re

class predict:
    def __init__(self, inputNodes, hiddenNodes, outputNodes, learningRate, rnnSize):
        self.iNodes = inputNodes
        self.hNodes = hiddenNodes
        self.oNodes = outputNodes
        self.lr = learningRate
        self.rs = rnnSize
        self.w = []
        for i in range(self.rs + 1):
            if i == 0:
                self.w.append(numpy.random.normal(0.0, pow(self.hNodes, -0.5), (self.hNodes, self.oNodes)))
                continue
            if i == self.rs:
                self.w.append(numpy.random.normal(0.0, pow(self.oNodes, -0.5), (self.oNodes, self.hNodes)))
                continue
            # 정규 분포의 (중심은 0.0, 표준 편차 연결 노드에 들어오는 개수 루트 역수, numpy 행렬)
            self.w.append(numpy.random.normal(0.0, pow(self.hNodes, -0.5), (self.hNodes, self.hNodes)))

        # 활성화 함수
        self.af = lambda x: scipy.special.expit(x)

    def train(self, source, target):
        source = numpy.array(source, ndmin=2).T
        target = numpy.array(target, ndmin=2).T
        outputs = []
        # 정방향
        for i in range(self.rs +1):
            if i == 0:
                inputs = numpy.dot(self.w[i],source)
                outputs.append(self.af(inputs))
                continue
            inputs = numpy.dot(self.w[i], outputs[i-1])
            outputs.append(self.af(inputs))
        # print(outputs)
        # print(target)

        # 역전파 오차
        errors = []
        for i in range(self.rs +1):
            errors.append('')
        for i in range(self.rs +1):
            i = self.rs - i
            if i == self.rs:
                errors[i] = target - outputs[i]
                continue
            errors[i] = numpy.dot(self.w[i+1].T, errors[i+1])

        # print(errors)

        # 갱신
        for i in range(self.rs +1):
            if i ==0:
                self.w[i] += self.lr * numpy.dot(errors[i] * outputs[i] * (1.0 - outputs[i]), numpy.transpose(source))
                continue
            self.w[i] += self.lr * numpy.dot(errors[i] * outputs[i] * (1.0 - outputs[i]), numpy.transpose(outputs[i-1]))

    def test(self, inputs):
        inputs = numpy.array(inputs, ndmin=2).T
        output = []
        for i in range(self.rs + 1):
            if i == 0:
                input = numpy.dot(self.w[i], inputs)
                output = self.af(input)
                continue
            input = numpy.dot(self.w[i], output)
            output = self.af(input)
        # print(output.flatten().argsort())
        return output.flatten().argsort()

if __name__ == "__main__":
    inputNodes = 45
    hiddenNodes = 45
    outputNodes = 45
    learningRate = 0.3
    rnnSize = 2
    p = predict(inputNodes, hiddenNodes, outputNodes, learningRate, rnnSize)

    # 트레이닝
    f = open('train_data.txt', 'r')
    lines = f.readlines()
    epochs = 100
    for e in range(epochs):
        for line in lines:
            line= re.sub('\n','', line)
            line = line.split(',')
            inputs = numpy.zeros(inputNodes) + 0.01
            outputs = numpy.zeros(outputNodes) + 0.01
            for i in range(7,13):
                outputs[int(line[i])-1] = 0.99
            for i in range(6):
                inputs[int(line[i])-1] = 0.99
            p.train(inputs, outputs)

    # 예측률
    f = open('test_data.txt', 'r')
    lines = f.readlines()
    lineLen = len(lines)
    prob = []
    for i in range(7):
        prob.append(0)
    for line in lines:
        line = re.sub('\n', '', line)
        line = line.split(',')
        inputs = numpy.zeros(inputNodes) + 0.01
        outputs = []
        for i in range(7, 13):
            outputs.append(line[i])
        for i in range(6):
            inputs[int(line[i]) - 1] = 0.99
        result = p.test(inputs)
        score = 0
        for i in range(1,7):
            if(str(int(result[-i]) + 1)) in outputs:
                score += 1
        prob[score] += 1/lineLen

    for i in range(7):
        print(str(i) + '개 맞을 확률 : ' + str(prob[i]))

    # value = [4,7,13,29,31,39]
    value = [5,11,12,29,33,44]
    for i in range(6):
        inputs[value[i]] = 0.99
    result = p.test(inputs)
    print(result)
    for i in range(45):
        result[i] +=1
    print(result)
    # for i in range(1,7):
    #     a = int(result[-i]) + 1
    #     print(a)
