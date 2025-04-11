import numpy as np
import itertools

class Parameters:
    @staticmethod
    def generate_parameters(len_x, lb,ub):
        elements = [0, 1]
        dict_parameters = {"x1": {(): np.random.uniform(lb, ub)}}
        for k in range(2, len_x + 1):
            permutations = list(itertools.product(elements, repeat=k - 1))
            dict_parameters["x" + str(k)] = {perm: np.random.uniform(lb, ub) for perm in permutations}
        return dict_parameters




class Distribution(object):
    def __init__(self, num_variables, initial_parameters=None):
        self.num_variables = num_variables
        
        self.parameters = initial_parameters if initial_parameters is not None else Parameters().generate_parameters(num_variables,lb=0,ub=2.0)
        print(self.parameters)
        self.learning_rate = 0.001

 
    def forward_process(self, dict_parameters):
        elements = [0, 1]
        dict_new_parameters = {"x1": {(): self.sigmoid(dict_parameters["x1"][()])}}
        for k in range(2, self.num_variables + 1):
            permutations = list(itertools.product(elements, repeat=k - 1))
            dict_new_parameters["x" + str(k)] = {perm: self.sigmoid(dict_parameters["x" + str(k)][perm]) for perm in permutations}
        return dict_new_parameters

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, sig):
        return sig * (1 - sig)

    def _generate_prob_sample(self, prob_dist):
        list_sample, prob = [], []
        for i in range(1, len(prob_dist) + 1):
            p = prob_dist["x" + str(i)][tuple(list_sample)]
            sample = np.random.binomial(1, p)
            prob.append(p if sample == 1 else 1 - p)
            list_sample.append(sample)
        return prob, list_sample

    def logp_derivative(self, sample_X):
        deriv_sums = self._initialize_derivatives()
        prob_dist = self.forward_process(self.parameters)
        
        for sample in sample_X:
            # For each sample, calculate the gradient of log likelihood
            for i in range(1, len(sample) + 1):
                cond_var = tuple(sample[:i - 1])
                p = prob_dist["x" + str(i)][cond_var]
                

                # Adjust gradient based on sample outcome (1 or 0)
                #grad =   p * (1 - p) * (sample[i - 1] - p) / (p * (1 - p) + 1e-10)

                grad = (1-p) if sample[i-1]==1 else -p 
                #self.sigmoid_derivative(p) * (sample[i - 1] - p) / (p * (1 - p) + 1e-10)
                deriv_sums["dx" + str(i)][cond_var] += grad
        
        # Average gradients over all samples
        for k in deriv_sums:
            for key in deriv_sums[k]:
                deriv_sums[k][key] /= len(sample_X)
        
        return deriv_sums

    def _initialize_derivatives(self):
        elements = [0, 1]
        dict_derivs = {"dx1": {(): 0.0}}
        for k in range(2, self.num_variables + 1):
            permutations = list(itertools.product(elements, repeat=k - 1))
            dict_derivs["dx" + str(k)] = {perm: 0.0 for perm in permutations}
        return dict_derivs

    def generate_samples(self, num_samples):
        list_samples = []
        prob_dist = self.forward_process(self.parameters)  # Use updated parameter probabilities
        for _ in range(num_samples):
            prob, sample = self._generate_prob_sample(prob_dist=prob_dist)
            prob_product = round(np.prod(prob), 6)
            list_samples.append((sample, prob_product))  # Store both sample and its probability
        return list_samples
    
    def update_parameters(self, deriv_sums):
        self.parameters["x1"][()] += self.learning_rate * deriv_sums["dx1"][()]
        for i in range(2, self.num_variables + 1):
            for perm in self.parameters["x" + str(i)]:
                self.parameters["x" + str(i)][perm] += self.learning_rate * deriv_sums["dx" + str(i)][perm]

    def log_likelihood(self, sample_X):
        log_likelihood_sum = 0.0
        prob_dist = self.forward_process(self.parameters)
        for sample in sample_X:
            sample_prob = self.get_sample_prob(sample, prob_dist)
            log_likelihood_sum += np.log(sample_prob + 1e-10)
        return log_likelihood_sum

    def get_sample_prob(self, sample, prob_dist):
        prob = 1.0
        for i in range(1, len(sample) + 1):
            p = prob_dist["x" + str(i)][tuple(sample[:i - 1])]
            prob *= p if sample[i - 1] == 1 else (1 - p)
        return prob

    def kl_divergence(self, sample_X, other):
        kl_sum = 0.0
        epsilon = 1e-10  # Small constant to avoid division by zero
        self_prob_dist = self.forward_process(self.parameters)
        other_prob_dist = other.forward_process(other.parameters)
        
        for sample in sample_X:
            self_prob = self.get_sample_prob(sample, prob_dist=self_prob_dist)
            other_prob = other.get_sample_prob(sample, prob_dist=other_prob_dist)
            
            # Add epsilon to both probabilities to handle log(0) cases
            kl_sum += self_prob * np.log((self_prob + epsilon) / (other_prob + epsilon))
        
        return kl_sum
 


initial_parameters = Parameters().generate_parameters(len_x=5, lb=0.0, ub=3.0)

print(initial_parameters)
batch_size = 1000 
print("Training Loop Start's Here ::")
# Original Distribution
num_variables = 5
# p(x1, x2, x3, x4, x5) = p(x1) p(x2|x1) p(x3|x2,x1) ...




# initial_parameters = {'x1': {(): np.float64(0.6105838933937734)}, 
#                       'x2': {(0,): np.float64(1.7219534503097305), (1,): np.float64(2.120384662660463)}, 
#                       'x3': {(0, 0): np.float64(1.205721147176752), (0, 1): np.float64(1.7663034765656154), (1, 0): np.float64(1.5868785797181886), (1, 1): np.float64(2.2475606318724775)},
#                       'x4': {(0, 0, 0): np.float64(0.8718638681461294), (0, 0, 1): np.float64(1.3777744788928312), (0, 1, 0): np.float64(1.3872575613238112), (0, 1, 1): np.float64(1.6344756295373246), (1, 0, 0): np.float64(1.2283292311493113), (1, 0, 1): np.float64(1.3844796344236854), (1, 1, 0): np.float64(1.9171894239876563), (1, 1, 1): np.float64(1.8072898440282397)},
#                       'x5': {(0, 0, 0, 0): np.float64(0.6594719945861921), (0, 0, 0, 1): np.float64(0.8885058162122675), (0, 0, 1, 0): np.float64(0.6757744234817903), (0, 0, 1, 1): np.float64(1.0225089642190064), (0, 1, 0, 0): np.float64(0.6843202607735592), (0, 1, 0, 1): np.float64(0.8031412634690331), (0, 1, 1, 0): np.float64(1.1444794752930405), (0, 1, 1, 1): np.float64(1.1141948522569136), (1, 0, 0, 0): np.float64(1.083513235037557), (1, 0, 0, 1): np.float64(0.6979658130989729), (1, 0, 1, 0): np.float64(0.9058050137043484), (1, 0, 1, 1): np.float64(1.1277455484666434), (1, 1, 0, 0): np.float64(1.13679609761262), (1, 1, 0, 1): np.float64(1.6394609678500875), 
#                              (1, 1, 1, 0):
#                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     np.float64(1.288949071159566), (1, 1, 1, 1): np.float64(1.4528234487164917)}}
# #initial_parameters = {'x1': {(): np.float64(0.9)}, 'x2': {(0,): np.float64(0.6), (1,): np.float64(1.2)}} 
ori_dist_obj = Distribution(num_variables=num_variables, initial_parameters=initial_parameters)
data_X = ori_dist_obj.generate_samples(num_samples=100000)
sample_X = [temp[0] for temp in data_X]

# Initialize a random distribution
learn_dist_obj = Distribution(num_variables=num_variables)

# Training loop with increased learning rate and gradient averaging
learn_dist_obj.learning_rate = 0.01  # Try a larger learning rate

batch_size = 1000
num_batches = len(sample_X) // batch_size
num_epochs = 500

for epoch in range(1, num_epochs + 1):
    print(f"\nEpoch {epoch}/{num_epochs}")
    
    for i in range(num_batches):
        sample_data = sample_X[i * batch_size: batch_size * (i + 1)]
        # Calculate and update gradients
        deriv_sums = learn_dist_obj.logp_derivative(sample_X=sample_data)
        learn_dist_obj.update_parameters(deriv_sums=deriv_sums)
        
        
    # Monitor learning progress each epoch
    curr_kl_div = ori_dist_obj.kl_divergence(sample_X=sample_X, other=learn_dist_obj)
    log_likelihood = learn_dist_obj.log_likelihood(sample_X=sample_X)
    #print(learn_dist_obj.parameters)
    print(f"KL Divergence: {curr_kl_div:.6f} | Log Likelihood: {log_likelihood:.6f}")

print("learned Distribution vs original distribution")

for key in ori_dist_obj.parameters:
    for key2 in ori_dist_obj.parameters[key]:
        print( f" key :: {key}   given :: {key2}   original :: { ori_dist_obj.parameters[key][key2]}  learned :: {learn_dist_obj.parameters[key][key2]}   "      )