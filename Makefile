.PHONY: ppo_train ppo_eval a2c_train a2c_eval dt_collect dt_train dt_eval qlearn_train qlearn_eval

ppo_train:
	-@ulimit -n 4096
	python src/ppo_train.py

ppo_eval:
	python src/ppo_eval.py --render

a2c_train:
	-@ulimit -n 4096
	python src/a2c_train.py

a2c_eval:
	python src/a2c_eval.py --render

dt_collect:
	python src/dt_collect.py

dt_train:
	python src/dt_train.py

dt_eval:
	python src/dt_eval.py --render

qlearn_train:
	python src/qlearning_train.py

qlearn_eval:
	python src/qlearning_eval.py --render