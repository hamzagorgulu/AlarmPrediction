import pickle 
import torch

with open("datasets/input_output.pkl", "rb") as f:
	input_output_dict = pickle.load(torch.load(f, map_location = torch.device("cpu")))

breakpoint()