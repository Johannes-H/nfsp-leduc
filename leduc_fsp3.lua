local fsp = require 'fsp'
local setup = require 'setup_fsp3'

local experiment_path = arg[1] or "."
local log_filename = experiment_path.."/eval.log"
local save_filename = experiment_path.."/saved"

print("Log file is "..log_filename)
print("Save file is "..save_filename)

local function generate_Full1(HUs)
  return function(nIn, nOut, softmax)
    return fsp.net.Full1(nIn, nOut, HUs, softmax)
  end
end


local params = {  log_file = log_filename,
                  plot_eval = false,
                  save_file = save_filename,
                  nPlayers = 2,
                  eval_freq = 1000,
                  save_freq = -1,  -- only latest
                  minReplayData = 1000,
                  rl_minibatch_size = 128,
                  sl_minibatch_size = 128,
                  budget_train_br = 2,
                  budget_train_avg = 2,
                  anticipatory_base = 0.1,
                  anticipatory_const = 0,
                  anticipatory_min = 0.1,
                  lr_base = {br = 1e-1, avg = 5e-3},
                  lr_const = {br = 0, avg = 0},
                  explo_base = 0.06,
                  explo_const = 1e-2,
		          gpu = false,
                  }
local config = {  simulator = fsp.simulator.Leduc,
                  evaluator = fsp.simulator.LeducEval,
                  createModel = generate_Full1({64}),
                  nGames = 128,
                  rl_mem_cap = 2e5,
                  sl_mem_cap = 2e6,
                  sl_mem_exp_beta_multiple = 0,
                  q_target_net_refresh = 0,
                  optim_method = {br = optim.sgd, avg = optim.sgd},
                  }


local algo = setup(params, config)

algo:run(1000000000)
