#https://www.zybuluo.com/hanbingtao/note/448086
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
		return x

	def getTrainDataSet(self):
		return [
			[(5,), 5500],
			[(3,), 2300],
			[(8,), 7600],
			[(1.4,), 1800],
			[(10.1,), 11400]
			];

o =  perception(1, 10, 0.01)
o.train()
print o
print 'work years: 1 = %.2f' % o.prediction([1,])
print 'work years: 2= %.2f' % o.prediction([2,])
print 'work years: 10 = %.2f' % o.prediction([10,])
print 'work years: 6.3 = %.2f' % o.prediction([6.3])
