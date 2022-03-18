# non contextual RL agent with InvertedResidual nn architecture

python -m carla.train_agent --norm_obs --seed 44864 --exp_name inv_res --hid_context_net 0 --hid_state_net -1 --steps 2048
python -m carla.train_agent --norm_obs --seed 26912 --exp_name inv_res --hid_context_net 0 --hid_state_net -1 --steps 2048


python -m carla.train_agent --norm_obs --seed 44864 --exp_name inv_res_nbn --hid_context_net 0 --hid_state_net -2 --steps 2048
python -m carla.train_agent --norm_obs --seed 26912 --exp_name inv_res_nbn --hid_context_net 0 --hid_state_net -2 --steps 2048

