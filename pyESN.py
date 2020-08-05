import numpy as np

# read in data
def get_data(filename, T, K, U, L, Y, TEST_LEN, TEST_IN, TEST_OUT):
	input_data = []
	teacher_data = []

	with open(filename, "r") as data:
		for line in data.readlines():
			line = line.split(" ")
			input_data.append(int(line[0])/255) # in range 0 -1
			teacher_data.append(int(line[1])/90) # in range 0 - 1

	# used for training
	for i in range(T):
		for n in range(K):
			U[i][n] = input_data[i+n]
		for n in range(L):
			Y[i][n] = teacher_data[i+n]
	
	# used for testing later
	for i in range(TEST_LEN):
		TEST_IN[i][:] = input_data[i + T: i + T + K]
		TEST_OUT[i][:] = teacher_data[i + T: i + T + L]

	return U, Y, TEST_IN, TEST_OUT

# procedure to ensure that W has the ESP
def initialize_W(sparsity, scaling_factor_alpha, W, N):
	# make the matrix sparse
	for x in range(N):
		for y in range(N):
			chance = np.random.random_sample()
			if chance <= sparsity:
				W[x][y] = np.random.uniform(-1.0, 1.0)

	max_eig_val = 0
	#  find the largest absolute eigenvalue (spectral_radius)
	for i in np.linalg.eigh(W)[0]:
		if i < 0:
			i *= -1
		if i > max_eig_val:
			max_eig_val = i

	# normalise and scale
	W0 = (1/max_eig_val) * W
	W = scaling_factor_alpha * W0
	
	return W

def generate_reservoir_state(X, N, Y_ACTUAL, L, T, W_in, U, W, W_back, Y, K, W_out):
	# X will store all the reservoir states at each time step
	X[0][:] = np.zeros((1, N))
	# Y_ACTUAL will store the outputs from the esn at each time step. To be compared against Y later
	Y_ACTUAL[0][:] = np.zeros((1, L))
	for i in range(T-1):
		new_x = np.tanh( (np.matmul(W_in, np.transpose(U[i+1][:]))) + (np.matmul(W, np.transpose(X[i][:]))) + (np.matmul(W_back, np.transpose(Y[i][:]))) ) # can add epsilon vector here for accuracy
		X[i + 1][:] = np.transpose(new_x)
		
		pre_output = np.empty((1, K+N+L))
		pre_output[0][:K] = U[i + 1][:]
		pre_output[0][K:K+N] = X[i + 1][:]
		pre_output[0][K+N:K+N+L] = Y_ACTUAL[i][:]
		output = np.tanh( (np.matmul(W_out, np.transpose(pre_output))) )
		Y_ACTUAL[i + 1][:] = np.transpose(output)

	return X, Y_ACTUAL

# the state collecting matrix
def collect_M(M, Y_ACTUAL, T0, T, K, U, N, X, L):
	for i in range(T - T0):
		M[i][:K] = U[i + T0][:]
		M[i][K:K+N] = X[i + T0][:]
		M[i][K+N:K+N+L] = Y_ACTUAL[(i-1) + T0][:]

	return M

# the teacher matrix (T in the original paper)
def collect_C(C, Y, T0, T):
	for i in range(T - T0):
		C[i][:] = np.arctanh(Y[i + T0][:])

	return C

# regression to train the outputs
def train_W_out(W_out, M, C):
	W_out = np.matmul(np.linalg.pinv(M), C)
	W_out = np.transpose(W_out)

	return W_out

def evaluate(step, x, y, W_in, TEST_IN, W, W_back):
	tmp = np.tanh( (np.matmul(W_in, np.transpose(TEST_IN[step][:]))) + np.matmul(W, x) + np.matmul(W_back, y) )
	return tmp

def exploit(step, x, y, K, N, L, TEST_IN, W_out):
	pre_output = np.empty((1, K+N+L))
	pre_output[0][:K] = TEST_IN[step][:]
	pre_output[0][K:K+N] = np.transpose(x)
	pre_output[0][K+N:K+N+L] = np.transpose(y)
	pre_output = np.transpose(pre_output)
	tmp = np.tanh( np.matmul(W_out, pre_output) )
	return tmp[0]

# returs the average absolute error
def RMSD(actual, outputted):
	total = len(actual)
	running_total = 0
	for i in range(total):
		tmp = ((actual[i] * 90) - (outputted[i] * 90)) ** 2
		tmp = tmp ** 0.5
		running_total += tmp
	return running_total/total


if __name__ == "__main__":

	"""
		Global definitions and matrices/vectors for ESN
	"""

	K = 1 # inputs
	N = 100 # reservoir size
	L = 1 # outputs
	SCALING_FACTOR = 0.8
	SPARSITY = 0.2 # W matrix density, max is usually about 0.25 - 0.3
	T = 2000 # training data steps
	T0 = 100 # to be discarded
	EPSILON = 0.002 # noise...not used here. can be added as a fourth term inside evaluation
	TEST_LEN = 500

	# Input data and teacher forcing matrices
	U = np.empty((T, K)) # input data u(n)
	Y = np.empty((T, L)) # for teacher forcing d(n)
	Y_ACTUAL = np.empty((T, L)) # exploited outputs during training...
	
	# for testing input and output after training
	TEST_IN = np.empty((TEST_LEN, K))
	TEST_OUT = np.empty((TEST_LEN, L)) # will be used to check accuracy
	TEST_OUT_ACTUAL = np.empty((TEST_LEN, L))
	
	# Weight matrices
	W_in = np.random.uniform(-1.0, 1.0, (N, K))
	W = np.zeros((N, N)) # will initialize later so esp is maintained
	W_out = np.random.uniform(-1.0, 1.0, (L, K+N+L))
	
	#W_back = np.random.uniform(-1.0, 1.0, (N, L)) # for active signal generation
	W_back = np.zeros((N, L)) # used for passive filtering
	
	# will collect  reservoir states
	X = np.empty((T, N))
	
	# state and teacher collecting matrices
	M = np.empty((T - T0, K+N+L))
	C = np.empty((T - T0, L))
	
	# vectors for current state and output
	x = np.zeros((N, 1))
	y = np.zeros((L, 1))
	
	U, Y, TEST_IN, TEST_OUT = get_data("pwm_to_odom_minvolt.txt", T, K, U, L, Y, TEST_LEN, TEST_IN, TEST_OUT)
	W = initialize_W(SPARSITY, SCALING_FACTOR, W, N)
	X, Y_ACTUAL = generate_reservoir_state(X, N, Y_ACTUAL, L, T, W_in, U, W, W_back, Y, K, W_out)
	M = collect_M(M, Y_ACTUAL, T0, T, K, U, N, X, L)
	C = collect_C(C, Y, T0, T)
	W_out = train_W_out(W_out, M, C)
	
	x = np.transpose(X[-1][:]) # the last state during training
	y = np.transpose(Y_ACTUAL[-1]) # the last output during training

	data_not_normal = np.empty((TEST_LEN, L)) # for the data in its original range
	for step in range(TEST_LEN):
		x = evaluate(step, x, y, W_in, TEST_IN, W, W_back)
		y = exploit(step, x, y, K, N, L, TEST_IN, W_out)
		TEST_OUT_ACTUAL[step][:] = np.transpose(y)
		data_not_normal[step][:] = np.transpose(y * 90)

	#print(TEST_OUT)

	error = RMSD(TEST_OUT, TEST_OUT_ACTUAL)
	print(data_not_normal)
	#print(W_out)
	print(error)