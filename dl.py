#https://www.zybuluo.com/hanbingtao/note/433855
class perception(object):
	def __init__(self, input_num, iteration, rate):
		self.iteration = iteration
		self.rate = rate
		self.weights = [0.0 for _ in range(input_num)]
		self.bias = 0.0

	def train(self):
		dataSet = self.getTrainDataSet()
		for i in range(self.iteration):
			self.oneIteration(dataSet)

	def oneIteration(self, dataSet):
		for input_vec, label in dataSet:
			output = self.prediction(input_vec)
			self.updateWeight(input_vec, label, output)


	def prediction(self, input_vec):
		temp = zip(input_vec, self.weights)
		temp1 = map(lambda (x,y): x * y, temp)
		return self.f(reduce(lambda x, y: x + y, temp1) + self.bias)

	def updateWeight(self, input_vec, label, output):
		diff = label - output
		self.weights = map(lambda (x, w): w + x * diff * self.rate, zip(input_vec, self.weights))
		self.bias += diff * self.rate

	def __str__(self):
		return 'weights: %s\nbias: %s' % (self.weights, self.bias)

	def f(self, x):
		return 1 if x > 0 else 0

	def getTrainDataSet(self):
		return [
			[(1, 0), 0],
			[(1, 1), 1],
			[(0, 0), 0],
			[(0, 1), 0]
			];

o =  perception(2, 5, 0.1)
o.train()
print o
print '1 and 1 = %d' % o.prediction([1, 1])
print '1 and 0 = %d' % o.prediction([1, 0])
print '0 and 1 = %d' % o.prediction([0, 1])
print '0 and 0 = %d' % o.prediction([0, 0])