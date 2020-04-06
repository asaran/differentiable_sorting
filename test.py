import torch
from differentiable_sorting.torch import bitonic_matrices, diff_sort, diff_argsort
from torch.autograd import Variable
import numpy as np

def sort(x):
	matrices = bitonic_matrices(len(x))
	torch_input = Variable(torch.from_numpy(np.array(x)).float(), requires_grad=True)
	sorted_input = diff_sort(matrices, torch_input)
	print('sorted_input: ',sorted_input)
	ranking = diff_argsort(matrices, torch_input)
	print('ranking: ',ranking)

	# differentiate the sorted array wrt input 
	print('derivative: ',torch.autograd.grad(sorted_input[0], torch_input)[0])

	return sorted_input, ranking

def LB_div(gt,r):
	# gt - ground truth ranking
	# r - scores from a reward network

	# sort reward scores r
	sorted_r, ranking = sort(r)
	#ranking = ranking + 1 #indexing with 1 instead of 0

	#gt = Variable(torch.from_numpy(np.array(gt)).float(), requires_grad=True)
	#r = Variable(torch.from_numpy(np.array(r)).float(), requires_grad=True)
	sorted_r = torch.tensor(sorted_r)
	r = torch.tensor(r)
	r_permuted = r.index_select(0, gt)
	print('scores permuted by GT: ',r_permuted)

	penalty = 0
	# iterate over trajectories
	for i in range(len(r)):
		# sorted scores, scores permuted acc to GT ranking (still non-differentiable)
		penalty += (sorted_r[i] - r_permuted[i])*i

	return penalty

if __name__ == '__main__':
	# sort an array x
	#x = [5.0, -1.0, 9.5, 13.2, 16.2, 20.5, 42.0, 18.0]
	#sort(x)

	## Simulate LB divergence penalty and check its differentiability
	print("Computing LB divergences...\n")

	# ground truth ranking of a set of trajectories
	# ranking signifies ascending order of rewards
	gt_ranking = torch.tensor([1,0,2,3])
	#gt_ranking = Variable(torch.from_numpy(np.array(gt_ranking)).float(), requires_grad=True)
	print("Ground Truth Ranking: "+str(gt_ranking))

	# scores sa from reward network A
	# should have high penalty as gives a low score to best trajectory (index 2, last traj)
	sa = [5, 20, 2, 25]
	#sa = Variable(torch.from_numpy(np.array(sa)).float(), requires_grad=True)
	print("Scores from reward network A: "+str(sa))
	lb_div_a = LB_div(gt_ranking, sa) 
	print("LB divergence for A: "+str(lb_div_a))

	print("\n")

	# scores from reward network B
	# should have low penalty as gives highest score to best trajectory
	sb = [2, 5, 20, 25]
	#sb = Variable(torch.from_numpy(np.array(sb)).float(), requires_grad=True)
	print("Scores from reward network B: "+str(sb))
	lb_div_b = LB_div(gt_ranking, sb)
	print("LB divergence for B: "+str(lb_div_b))

