require 'optim'
local fsp = require 'fsp'
local tablex = require 'pl.tablex'
torch.setdefaulttensortype('torch.FloatTensor')


local function createOptimState(params, hyper_params, policy)
  optim_state = {}
  optim_state.learningRate = params.lr_base[policy]
  optim_state.momentum = 0
  print("Using SGD without momentum for "..policy)
  return optim_state
end

local function create_agents(params, hyper_params, br_nets, avg_nets)
  local create_player_agents = function(p)
    return fsp.agent.CreateMultipleFSP{nAgents = hyper_params.nGames, mem_rl = params.rl_memories[p], mem_sl = params.sl_memories[p], anticipatory = params.anticipatory_min,
                                         light_br = fsp.policy.LightEpsGreedy{net = br_nets[p], epsilon = params.explo_base, actionDim = params.actionDim},
                                         light_avg = fsp.policy.LightNetwork{net = avg_nets[p]}}
  end
  return {create_player_agents(1), create_player_agents(2)}
end

local function setup(params, hyper_params)
  local createModel = hyper_params.createModel or error("Need to specify createModel function (nIn, nOut, softmax)")
  local simulator = hyper_params.simulator()
  local stateDim = simulator:stateDim()
  local actionDim = simulator:actionDim()
  params.stateDim = stateDim
  params.actionDim = actionDim
  print("StateDim: "..params.stateDim)
  params.rl_memories = {fsp.memory.WholeTransition({capacity = hyper_params.rl_mem_cap, stateDim = stateDim}), fsp.memory.WholeTransition({capacity = hyper_params.rl_mem_cap, stateDim = stateDim})}

  local sl_mem_exp_beta = hyper_params.sl_mem_cap * hyper_params.sl_mem_exp_beta_multiple
  params.sl_memories = {fsp.memory.ActionReservoir({capacity = hyper_params.sl_mem_cap, stateDim = stateDim, exp_beta = sl_mem_exp_beta}), fsp.memory.ActionReservoir({capacity = hyper_params.sl_mem_cap, stateDim = stateDim, exp_beta = sl_mem_exp_beta})}

  local avg_nets = {createModel(stateDim, actionDim, true), createModel(stateDim, actionDim, true)}
  params.avg_policies = {fsp.policy.Network({net = avg_nets[1]}), fsp.policy.Network({net = avg_nets[2]})}
  local br_nets = {createModel(stateDim, actionDim), createModel(stateDim, actionDim)}
  params.br_policies = {fsp.policy.Greedy({net = br_nets[1], actionDim = actionDim}), fsp.policy.Greedy({net = br_nets[2], actionDim = actionDim})}

  local agents = create_agents(params, hyper_params, br_nets, avg_nets)
  local agents_forward = function(batch_o_t)
    for p=1,2 do
      avg_nets[p]:forward(batch_o_t)
      br_nets[p]:forward(batch_o_t)
    end
  end
  params.dataGen = fsp.data.ZeroSumGenerator({simulator = simulator, nGames = hyper_params.nGames, agents = agents, agents_forward = agents_forward, stateDim = stateDim})

  local optim_state
  params.br_learners = {fsp.learner.Q({net = br_nets[1], avg = params.avg_policies[1], target_net_refresh = hyper_params.q_target_net_refresh, optim_method = hyper_params.optim_method.br, optim_state = createOptimState(params, hyper_params, 'br'), gpu = params.gpu}),
                        fsp.learner.Q({net = br_nets[2], avg = params.avg_policies[2], target_net_refresh = hyper_params.q_target_net_refresh, optim_method = hyper_params.optim_method.br, optim_state = createOptimState(params, hyper_params, 'br'), gpu = params.gpu})}
  params.avg_learners = {fsp.learner.AvgActions({net = avg_nets[1], optim_method = hyper_params.optim_method.avg, optim_state = createOptimState(params, hyper_params, 'avg'), gpu = params.gpu}),
                        fsp.learner.AvgActions({net = avg_nets[2], optim_method = hyper_params.optim_method.avg, optim_state = createOptimState(params, hyper_params, 'avg'), gpu = params.gpu})}
  params.evaluator = hyper_params.evaluator
  return fsp.algo.FSP3(params)
end

return setup
